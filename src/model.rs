// model.rs
use candle_core::{Tensor, Result};
use candle_nn::{Embedding, VarBuilder, Dropout, Module};
use crate::config::ModelConfig;
use crate::transformer_block::TransformerBlock;
use crate::rmsnorm::RmsNorm;

pub struct MinimalLLM {
    token_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    norm: RmsNorm,
    dropout: Dropout,
    d_model: usize,
}

impl MinimalLLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("token_embedding"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(TransformerBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }

        let norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            token_embedding, blocks, norm,
            dropout: Dropout::new(cfg.dropout),
            d_model: cfg.d_model,
        })
    }

    /// tokens: (batch, seq_len) i64 -> logits (batch, seq_len, vocab_size)
    pub fn forward(&self, tokens: &Tensor, train: bool) -> Result<Tensor> {
        let mut x = self.token_embedding.forward(tokens)?;
        x = (x * (self.d_model as f64).sqrt())?;
        x = self.dropout.forward(&x, train)?;

        for block in &self.blocks {
            x = block.forward(&x, train)?;
        }

        x = self.norm.forward(&x)?;
        x = self.dropout.forward(&x, train)?;

        // tied lm_head: x @ embedding.weight^T
        let emb_weight = self.token_embedding.embeddings(); // (vocab_size, d_model)
        x.broadcast_matmul(&emb_weight.t()?)
    }
}