// feedforward.rs
use candle_core::{Tensor, Result};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias, Dropout};

pub struct SwiGLUFeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    dropout: Dropout,
}

impl SwiGLUFeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f32, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(d_model, d_ff, vb.pp("gate_proj"))?;
        let up_proj   = linear_no_bias(d_model, d_ff, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(d_ff, d_model, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj, up_proj, down_proj,
            dropout: Dropout::new(dropout_p),
        })
    }

    /// x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        let hidden = self.dropout.forward(&hidden, train)?;
        self.down_proj.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};
    use candle_nn::VarBuilder;

    #[test]
    fn test_silu_value() {
        let device = Device::Cpu;
        let x = Tensor::new(&[1f32], &device).unwrap();
        let out = candle_nn::ops::silu(&x).unwrap();
        let val: f32 = out.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        // silu(1.0) = 1 * sigmoid(1) ≈ 0.7311
        assert!((val - 0.7311).abs() < 1e-3);
    }

    #[test]
    fn test_swiglu_shapes() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let ffn = SwiGLUFeedForward::new(4, 8, 0.0, vb.pp("ffn")).unwrap();
        let x = Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap();
        let out = ffn.forward(&x, false).unwrap();
        assert_eq!(out.dims(), &[1, 2, 4]);
    }
}