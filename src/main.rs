mod config;
mod tensor_utils;
mod rope;
mod rmsnorm;
mod attention;
mod feedforward;
mod transformer_block;
mod model;
mod train;
mod inference;
mod data;

use candle_core::{Device, DType, Tensor};
use candle_nn::{VarMap, VarBuilder};
use config::ModelConfig;
use model::MinimalLLM;

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    let cfg = ModelConfig {
        d_model: 384,
        n_heads: 8,
        n_kv_heads: 4,
        n_layers: 6,
        d_ff: 1536,
        max_seq_len: 512,
        rms_norm_eps: 1e-6,
        dropout: 0.0, // disable for inference smoke test
        attention_bias: false,
        vocab_size: 49152,
        // existing unused fields - keep for now
        muon_lr: 0.01,
        weight_decay: 0.1,
        max_steps: 2000,
        gradient_accumulation_steps: 4,
        eval_every: 500,
        eval_steps: 100,
    };

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    println!("Building model...");
    let model = MinimalLLM::new(&cfg, vb)?;
    println!("Model built successfully.");

    // Dummy input: batch=1, seq_len=8, random token ids in [0, vocab_size)
    let seq_len = 8;
    let dummy_tokens: Vec<u32> = (0..seq_len).map(|i| i * 37 % cfg.vocab_size as u32).collect();
    let input = Tensor::from_vec(dummy_tokens, (1, seq_len as usize), &device)?;

    println!("Running forward pass...");
    let logits = model.forward(&input, false)?;
    println!("Output logits shape: {:?}", logits.dims());

    // sanity: shape should be (1, seq_len, vocab_size)
    assert_eq!(logits.dims(), &[1, seq_len as usize, cfg.vocab_size]);
    println!("End-to-end forward pass successful on {:?}", device);

    Ok(())
}