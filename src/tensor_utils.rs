// tensor_utils.rs
use candle_core::{Tensor, Result};

pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;

    // (b, n_kv_heads, seq_len, head_dim) -> (b, n_kv_heads, 1, seq_len, head_dim)
    let x = x.unsqueeze(2)?;
    // expand to (b, n_kv_heads, n_rep, seq_len, head_dim)
    let x = x.expand((b, n_kv_heads, n_rep, seq_len, head_dim))?;
    // reshape to (b, n_kv_heads * n_rep, seq_len, head_dim)
    x.reshape((b, n_kv_heads * n_rep, seq_len, head_dim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    fn test_repeat_kv() -> Result<()> {
        let device = &Device::Cpu;
        
        // 1. Create a small tensor matching your strategy: (b=1, n_kv_heads=2, seq_len=2, head_dim=2)
        // Values: 0.0 to 7.0 (Total 8 elements)
        let x = Tensor::arange(0f32, 8f32, device)?
            .reshape((1, 2, 2, 2))?;
        
        // 2. Execute the function with n_rep = 2
        let n_rep = 2;
        let out = repeat_kv(&x, n_rep)?;
        
        // 3. Assert target dimensions: (1, 4, 2, 2)
        assert_eq!(out.dims(), &[1, 4, 2, 2]);

        // 4. Convert to vectors to assert the exact values match the repetition strategy
        // Original x layout across heads:
        // Head 0: [[0, 1], [2, 3]]
        // Head 1: [[4, 5], [6, 7]]
        let orig_head_0 = x.i((0, 0, .., ..))?.to_vec2::<f32>()?;
        let orig_head_1 = x.i((0, 1, .., ..))?.to_vec2::<f32>()?;

        // Repeated output layout across heads:
        // Head 0 (rep 1 of orig 0)
        // Head 1 (rep 2 of orig 0)
        // Head 2 (rep 1 of orig 1)
        // Head 3 (rep 2 of orig 1)
        let out_head_0 = out.i((0, 0, .., ..))?.to_vec2::<f32>()?;
        let out_head_1 = out.i((0, 1, .., ..))?.to_vec2::<f32>()?;
        let out_head_2 = out.i((0, 2, .., ..))?.to_vec2::<f32>()?;
        let out_head_3 = out.i((0, 3, .., ..))?.to_vec2::<f32>()?;

        // 5. Assert data equality matching your strategy rules
        assert_eq!(out_head_0, orig_head_0);
        assert_eq!(out_head_1, orig_head_0); // result[:,0,:,:] == result[:,1,:,:] == original[:,0,:,:]
        
        assert_eq!(out_head_2, orig_head_1);
        assert_eq!(out_head_3, orig_head_1); // Check second pair of KV repetitions

        Ok(())
    }
}