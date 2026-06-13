// main.rs
mod config; mod tensor_utils; mod rope; mod rmsnorm;
mod attention; mod feedforward; mod transformer_block; mod model;
mod train; mod inference; mod data;

use candle_core::{Device, DType};
use candle_nn::VarMap;
use config::ModelConfig;
use model::MinimalLLM;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;

    let cfg = ModelConfig {
        d_model: 384, n_heads: 8, n_kv_heads: 4, n_layers: 6,
        d_ff: 1536, max_seq_len: 512, rms_norm_eps: 1e-6,
        dropout: 0.1, attention_bias: false,
        vocab_size: 49152, // SmolLM tokenizer vocab size — confirm exact value
    };

    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MinimalLLM::new(&cfg, vb)?;

    // load data, call train::train_model(...)
    Ok(())
}