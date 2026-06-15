// rmsnorm.rs
#![allow(unused_imports)]
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

// rmsnorm.rs - replace the test
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn test_rmsnorm_manual() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RmsNorm::new(4, 1e-6, vb.pp("test")).unwrap();
        // weight now correctly initializes to ones via init::ONE hint

        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4), &device).unwrap();
        let out = norm.forward(&x).unwrap();

        // manual: rms = sqrt(mean([1,4,9,16]) + eps) = sqrt(7.5) ≈ 2.7386
        let expected: Vec<f32> = vec![1., 2., 3., 4.].iter().map(|v| v / 2.7386f32).collect();
        let out_vec: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();

        for (e, o) in expected.iter().zip(out_vec.iter()) {
            assert!((e - o).abs() < 1e-3, "expected {e}, got {o}");
        }
    }
}