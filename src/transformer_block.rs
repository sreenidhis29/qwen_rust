// transformer_block.rs
use candle_core::{Tensor, Result};
use candle_nn::{VarBuilder, Dropout};
use crate::config::ModelConfig;
use crate::attention::Qwen3Attention;
use crate::feedforward::SwiGLUFeedForward;
use crate::rmsnorm::RmsNorm;

pub struct TransformerBlock {
    attention: Qwen3Attention,
    feed_forward: SwiGLUFeedForward,
    norm1: RmsNorm,
    norm2: RmsNorm,
    dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Qwen3Attention::new(cfg, vb.pp("attention"))?,
            feed_forward: SwiGLUFeedForward::new(cfg.d_model, cfg.d_ff, cfg.dropout, vb.pp("feed_forward"))?,
            norm1: RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("norm1"))?,
            norm2: RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("norm2"))?,
            dropout: Dropout::new(cfg.dropout),
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let attn_out = self.attention.forward(&self.norm1.forward(x)?)?;
        let x = (x + self.dropout.forward(&attn_out, train)?)?;

        let ff_out = self.feed_forward.forward(&self.norm2.forward(&x)?, train)?;
        x + self.dropout.forward(&ff_out, train)?
    }
}