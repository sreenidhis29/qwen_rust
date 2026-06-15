// inference.rs
#![allow(dead_code, unused_imports)]
use candle_core::{Tensor, Result, D, DType};
use candle_nn::ops::softmax_last_dim;
use rand::distributions::{WeightedIndex, Distribution};
use rand::thread_rng;
use crate::model::MinimalLLM;

pub fn generate(
    model: &MinimalLLM,
    mut tokens: Vec<u32>,
    max_length: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    eos_token_id: u32,
    device: &candle_core::Device,
) -> Result<Vec<u32>> {
    for _ in 0..max_length {
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&input, false)?; // (1, t, vocab)
        let t = logits.dim(1)?;
        let last = logits.narrow(1, t - 1, 1)?.squeeze(1)?.squeeze(0)?; // (vocab,)

        let logits_vec: Vec<f32> = (last / (temperature as f64))?.to_vec1()?;
        let mut logits_vec = logits_vec;

        // Top-k
        if top_k > 0 && top_k < logits_vec.len() {
            let mut sorted: Vec<f32> = logits_vec.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[top_k - 1];
            for v in logits_vec.iter_mut() {
                if *v < threshold { *v = f32::NEG_INFINITY; }
            }
        }

        // Softmax
        let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits_vec.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<f32> = exp.iter().map(|&v| v / sum).collect();

        // Top-p (nucleus)
        if top_p < 1.0 {
            let mut idx: Vec<usize> = (0..probs.len()).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let mut cumsum = 0.0;
            let mut cutoff = idx.len();
            for (pos, &i) in idx.iter().enumerate() {
                cumsum += probs[i];
                if cumsum > top_p { cutoff = pos + 1; break; }
            }
            for &i in idx.iter().skip(cutoff) {
                probs[i] = 0.0;
            }
            let renorm: f32 = probs.iter().sum();
            for p in probs.iter_mut() { *p /= renorm; }
        }

        // Sample
        let dist = WeightedIndex::new(&probs).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let next_token = dist.sample(&mut thread_rng()) as u32;

        tokens.push(next_token);
        if next_token == eos_token_id { break; }
    }
    Ok(tokens)
}