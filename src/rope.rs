// rope.rs
use candle_core::{Tensor, Device, DType, Result};

pub struct Rotary {
    cos: Tensor, // (max_seq_len, head_dim/2)
    sin: Tensor,
}

impl Rotary {
    pub fn new(head_dim: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let half = head_dim / 4;
        // angular_freq: (1/10000)^(linspace(0,1,half))
        let exponents: Vec<f32> = (0..half)
            .map(|i| i as f32 / (half.max(1) as f32 - 1.0).max(1.0))
            .collect();
        let mut angular_freq: Vec<f32> = exponents.iter()
            .map(|&e| (1f32 / 10000f32).powf(e))
            .collect();
        // pad with zeros to head_dim/2
        angular_freq.extend(std::iter::repeat(0f32).take(head_dim / 2 - half));

        let angular_freq = Tensor::from_vec(angular_freq, head_dim / 2, device)?;
        let t = Tensor::arange(0f32, max_seq_len as f32, device)?;

        // theta[t,i] = t * angular_freq[i]  -> outer product
        let theta = t.unsqueeze(1)?.matmul(&angular_freq.unsqueeze(0)?)?; // (max_seq_len, head_dim/2)

        Ok(Self {
            cos: theta.cos()?,
            sin: theta.sin()?,
        })
    }

    /// x: (batch, seq_len, n_heads, head_dim)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, t, _h, d) = x.dims4()?;
        let half = d / 2;

        // slice cos/sin to seq_len, reshape for broadcast: (1, t, 1, half)
        let cos = self.cos.narrow(0, 0, t)?.reshape((1, t, 1, half))?;
        let sin = self.sin.narrow(0, 0, t)?.reshape((1, t, 1, half))?;

        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let y1 = (x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?)?;
        let y2 = (x2.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&y1, &y2], 3)
    }
}