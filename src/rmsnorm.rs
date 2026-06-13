// rmsnorm.rs
use candle_core::{Tensor, Device, DType, Result, Var};
use candle_nn::VarBuilder;

pub struct RmsNorm {
    weight: Tensor, // (dim,)
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::init::ONE)?;
        Ok(Self { weight, eps })
    }

    /// x: (..., dim) — normalizes over last dimension
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        let mean_sq = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&rms)?;

        let out = normed.to_dtype(orig_dtype)?;
        out.broadcast_mul(&self.weight)
    }
}