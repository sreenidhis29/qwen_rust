// data.rs
#![allow(dead_code)]
use candle_core::{Tensor, Device, Result};
use std::fs;

pub struct TextTokenDataset {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl TextTokenDataset {
    pub fn from_bin(path: &str, seq_len: usize) -> std::io::Result<Self> {
        let bytes = fs::read(path)?;
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0],c[1],c[2],c[3]]))
            .collect();
        Ok(Self { tokens, seq_len })
    }

    pub fn len(&self) -> usize {
        self.tokens.len().saturating_sub(self.seq_len)
    }

    pub fn get(&self, idx: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let x: Vec<u32> = self.tokens[idx..idx+self.seq_len].to_vec();
        let y: Vec<u32> = self.tokens[idx+1..idx+self.seq_len+1].to_vec();
        let x = Tensor::from_vec(x, self.seq_len, device)?.unsqueeze(0)?;
        let y = Tensor::from_vec(y, self.seq_len, device)?;
        Ok((x, y))
    }
}