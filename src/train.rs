// train.rs
use candle_core::{Tensor, Result, DType, D};
use candle_nn::{AdamW, ParamsAdamW, Optimizer, loss::cross_entropy};
use crate::config::ModelConfig;
use crate::model::MinimalLLM;

pub fn lr_lambda(step: usize, warmup_steps: usize, max_steps: usize) -> f64 {
    if step < warmup_steps {
        step as f64 / warmup_steps as f64
    } else {
        let progress = (step - warmup_steps) as f64 / (max_steps - warmup_steps) as f64;
        0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    }
}

pub fn train_step(
    model: &MinimalLLM,
    optimizer: &mut AdamW,
    x: &Tensor,
    y: &Tensor,
    vocab_size: usize,
    grad_accum_steps: usize,
) -> Result<f64> {
    let logits = model.forward(x, true)?; // (b, t, vocab)
    let (b, t, v) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, v))?;
    let targets_flat = y.reshape((b * t,))?;

    let loss = cross_entropy(&logits_flat, &targets_flat)?;
    let loss_scaled = (loss.clone() / grad_accum_steps as f64)?;

    let grads = loss_scaled.backward()?;
    optimizer.step(&grads)?; // candle accumulates if you manage grads externally; see note below

    loss.to_scalar::<f32>().map(|v| v as f64)
}

pub fn train_model(
    model: &MinimalLLM,
    cfg: &ModelConfig,
    train_data: &[(Tensor, Tensor)],
    val_data: &[(Tensor, Tensor)],
    varmap: &candle_nn::VarMap,
) -> Result<()> {
    let params = ParamsAdamW {
        lr: cfg.muon_lr * 0.1, // reuse field name; rename to `lr` in config
        weight_decay: cfg.weight_decay,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(varmap.all_vars(), params)?;

    let warmup_steps = cfg.max_steps / 20;
    let mut best_val_loss = f64::INFINITY;

    for step in 0..cfg.max_steps {
        let (x, y) = &train_data[step % train_data.len()];

        // update LR
        let lr = optimizer.learning_rate() * 0.0 + (cfg.muon_lr * 0.1) * lr_lambda(step, warmup_steps, cfg.max_steps);
        optimizer.set_learning_rate(lr);

        let loss = train_step(model, &mut optimizer, x, y, cfg.vocab_size.unwrap(), cfg.gradient_accumulation_steps)?;

        if step % 10 == 0 {
            println!("step {step}: loss={loss:.4}, lr={lr:.2e}");
        }

        if step % cfg.eval_every == 0 && step > 0 {
            let val_loss = evaluate(model, val_data, cfg)?;
            println!("step {step}: val_loss={val_loss:.4}");
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                varmap.save("best_model.safetensors")?;
            }
        }
    }

    varmap.save("final_model.safetensors")?;
    Ok(())
}

pub fn evaluate(model: &MinimalLLM, val_data: &[(Tensor, Tensor)], cfg: &ModelConfig) -> Result<f64> {
    let mut total_loss = 0.0;
    let mut total_tokens = 0usize;
    let n = cfg.eval_steps.min(val_data.len());

    for (x, y) in val_data.iter().take(n) {
        let logits = model.forward(x, false)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let targets_flat = y.reshape((b * t,))?;
        let loss = cross_entropy(&logits_flat, &targets_flat)?;
        total_loss += loss.to_scalar::<f32>()? as f64 * (b * t) as f64;
        total_tokens += b * t;
    }

    Ok(total_loss / total_tokens as f64)
}