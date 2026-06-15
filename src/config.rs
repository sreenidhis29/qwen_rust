// config.rs
#![allow(dead_code, unused_variables)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f64,
    pub dropout: f32,
    pub attention_bias: bool,
    pub vocab_size: usize,
    // training/eval params
    pub muon_lr: f64,
    pub weight_decay: f64,
    pub max_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub eval_every: usize,
    pub eval_steps: usize,
}

impl ModelConfig {
    pub fn d_k(&self) -> usize {
        self.d_model / self.n_heads
    }
    pub fn n_kv_groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}