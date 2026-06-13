// attention.rs
use candle_core::{Tensor, Device, DType, Result, D};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias, linear};
use crate::config::ModelConfig;
use crate::rope::Rotary;
use crate::rmsnorm::RmsNorm;
use crate::tensor_utils::repeat_kv;

pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    w_o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: Rotary,
    n_heads: usize,
    n_kv_heads: usize,
    n_kv_groups: usize,
    d_k: usize,
}

impl Qwen3Attention {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let d_k = cfg.d_k();
        let q_proj = if cfg.attention_bias {
            linear(cfg.d_model, cfg.n_heads * d_k, vb.pp("q_proj"))?
        } else {
            linear_no_bias(cfg.d_model, cfg.n_heads * d_k, vb.pp("q_proj"))?
        };
        let k_proj = linear_no_bias(cfg.d_model, cfg.n_kv_heads * d_k, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.d_model, cfg.n_kv_heads * d_k, vb.pp("v_proj"))?;
        let w_o = linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("w_o"))?;

        let q_norm = RmsNorm::new(d_k, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(d_k, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let rotary = Rotary::new(d_k, cfg.max_seq_len, vb.device())?;

        Ok(Self {
            q_proj, k_proj, v_proj, w_o, q_norm, k_norm, rotary,
            n_heads: cfg.n_heads, n_kv_heads: cfg.n_kv_heads,
            n_kv_groups: cfg.n_kv_groups(), d_k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self.q_proj.forward(x)?.reshape((b, t, self.n_heads, self.d_k))?;
        let k = self.k_proj.forward(x)?.reshape((b, t, self.n_kv_heads, self.d_k))?;
        let v = self.v_proj.forward(x)?.reshape((b, t, self.n_kv_heads, self.d_k))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE expects (b, t, heads, d_k) — matches our layout directly
        let q = self.rotary.forward(&q)?;
        let k = self.rotary.forward(&k)?;

        // -> (b, heads, t, d_k)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let k = repeat_kv(&k, self.n_kv_groups)?;
        let v = repeat_kv(&v, self.n_kv_groups)?;

        // scaled dot product, causal
        let scale = 1.0 / (self.d_k as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // (b, heads, t, t)

        let mask = causal_mask(t, x.device())?;
        let attn_scores = attn_scores.broadcast_add(&mask)?;

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let attn_out = attn_probs.matmul(&v)?; // (b, heads, t, d_k)

        let attn_out = attn_out.transpose(1, 2)?.contiguous()?
            .reshape((b, t, self.n_heads * self.d_k))?;

        self.w_o.forward(&attn_out)
    }
}

fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t * t)
        .map(|idx| {
            let i = idx / t;
            let j = idx % t;
            if j > i { f32::NEG_INFINITY } else { 0.0 }
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, t, t), device)
}