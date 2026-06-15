# qwen_rust

Qwen3-style decoder-only transformer architecture implemented from scratch in Rust using the Candle ecosystem (CUDA-enabled where available).

This repository contains a compact reimplementation of key architectural elements used in modern decoder-only LLMs, intended for research and experimentation. The code focuses on clarity and correctness rather than production performance.

## Highlights

- Qwen3-style decoder-only transformer implemented in Rust
- Features implemented:
	- Grouped Query Attention (GQA)
	- Rotary Positional Embeddings (RoPE)
	- QK-RMSNorm (RMS normalization applied to Q/K)
	- SwiGLU feed-forward network
- Forward pass verified end-to-end on GPU (Candle + CUDA)
- Training loop scaffolded (not a full train run in repo)

> Note: This repository implements the model architecture and a training scaffold. It does NOT include a pre-trained model. Numerical validation against PyTorch reference and a full training run are in progress.

## Repository structure

- `src/` — core Rust source files
	- `main.rs` — small smoke-test / example: builds model and runs a forward pass
	- `model.rs` — top-level model assembly (`MinimalLLM`)
	- `attention.rs` — attention implementation (GQA, RoPE, masking)
	- `rope.rs` — Rotary embedding support
	- `rmsnorm.rs` — RMS normalization module
	- `feedforward.rs` — SwiGLU FFN
	- `transformer_block.rs` — a single transformer block combining attention + FFN
	- `train.rs` — training loop scaffolding and helpers
	- `inference.rs` — generation utilities (sampling, top-k/top-p)
	- `data.rs` — lightweight dataset helpers
	- `tensor_utils.rs` — small tensor helpers
	- `config.rs` — model and training configuration
- `tests/` — unit and integration tests
- `Cargo.toml` — Rust manifest

## Quick start (development)

Prerequisites

- Rust toolchain (stable) and `cargo`
- CUDA toolkit and compatible GPU if you want to run CUDA-enabled backends

Build

```powershell
# build with default features (CPU)
cargo build

# or build with CUDA feature if Candle is configured for CUDA
cargo build --features cuda
```

Run the smoke test (forward pass)

```powershell
# run the example; this runs a small forward pass to verify shapes
cargo run --features cuda
```

If you do not have CUDA available, omit `--features cuda` and the code will run on CPU with Candle's CPU device.

## Tests

Run unit tests:

```powershell
cargo test
```

## Development notes

- Some modules contain scaffolding for training and evaluation that may be unused during a quick smoke-run; these are intentionally retained for iterative development and numerical validation. If you prefer strict linting, consider removing or using those symbols.
- The code currently uses selective `#![allow(...)]` attributes to reduce distracting warnings while preserving test/train code. These can be tightened later.

## License

See `LICENSE` in the repository root.

---