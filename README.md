# OAdamW

**OAdamW** is a hybrid optimizer that combines a StableAdamW-style adaptive update with optional **Muon (Newton–Schulz orthogonalised gradients)** for matrix parameters, along with optional Triton and foreach acceleration paths.

It is designed for stability in low precision training (fp16/bf16), while preserving high-rank gradient structure for weight matrices via orthogonalisation.

---

## Key Features

* StableAdamW-style optimizer core (adaptive moments + RMS stabilization)
* Optional **Muon mode** (Newton–Schulz orthogonalisation for 2D+ tensors)
* Debiased beta handling (no explicit bias-correction step)
* Kahan summation support for low-precision stability
* Foreach and Triton acceleration paths
* Fully decoupled weight decay option
* Gradient-release mode (fused backward step execution)
* Automatic model-aware parameter grouping

---

## What Makes This Different

### 1. Muon (Newton–Schulz Orthogonalisation)

Instead of compressing gradients (like SVD-based approaches), Muon applies **Newton–Schulz iteration** to push matrix gradients toward their closest orthogonal form.

* No rank reduction
* Same shape output
* Preserves full optimizer state structure
* Cheaper than SVD-based projection methods

This is applied only to 2D+ parameters (e.g. linear layers, embeddings).

---

### 2. StableAdamW Core

The optimizer builds on a StableAdamW-style update:

* EMA of gradients (β₁)
* EMA of squared gradients (β₂)
* RMS-based learning rate stabilization
* Decoupled or fully decoupled weight decay
* Optional Kahan summation for numerical stability

---

### 3. Performance Paths

* **Triton kernels** for per-tensor GPU execution
* **foreach kernels** for batched tensor updates
* Automatic fallback to safe single-tensor execution

---

## Installation

This optimizer depends on:

* PyTorch
* Triton (optional, for kernel acceleration)
* Optimi framework (`OptimiOptimizer` base class)

```bash
pip install torch triton
```

You also need the `optimi` package that provides the base optimizer infrastructure.

---

## Basic Usage

### Auto grouping (recommended)

```python
from oadamw import OAdamW

optimizer = OAdamW(model, lr=5e-6)
```

This automatically splits:

* 2D+ parameters → Muon path
* 1D parameters → standard AdamW path

---

### Manual parameter groups

```python
optimizer = OAdamW([
    {"params": linear_weights, "muon": True},
    {"params": biases},
], lr=5e-6)
```

---

## Muon Configuration

Muon groups support:

* `ns_steps` → Newton–Schulz iterations (default: 5)
* `update_proj_gap` → recompute interval (default: 1)
* `scale` → output scaling factor (default: 1.0)

Example:

```python
{"params": weights, "muon": True, "ns_steps": 6, "scale": 0.9}
```

---

## Design Notes

### Why Newton–Schulz instead of SVD?

SVD-based projection (used in methods like GaLore-style approaches):

* Expensive (O(n³) worst case)
* Requires rank selection
* Produces compressed gradients

Muon instead:

* Keeps full dimensionality
* Uses iterative matrix normalization
* Avoids decomposition overhead
* Integrates cleanly into Adam-style updates

---

### Why not fuse Muon into foreach/Triton?

Muon operates on **individual gradient tensors with matrix structure**, while foreach/Triton require homogeneous batched layouts.
Therefore Muon runs in a **single-parameter pre-processing path**, then delegates to the standard Adam core.

---

## Credits & Acknowledgements

This optimizer is a composition of ideas and implementations from multiple research and open-source contributions:

### Core inspiration

* **StableAdamW** — Benjamin Warner
  MIT License (2023–present)

* **Muon / Newton–Schulz orthogonalisation** — Kosson et al.
  Modular Muon codebase (2024)

* **Optimi framework** — Optimi project contributors
  Base optimizer infrastructure used throughout this implementation

---

### Numerical stability techniques

* **Kahan summation technique**
  Inspired by Meta’s `torchdistX` implementation:

  * AnyPrecisionAdamW (Meta Platforms, Inc.)

---

### GPU acceleration

* Triton kernel design inspiration:

  * AdamW-Triton-PyTorch — Less Wright (MIT License)
  * lion-pytorch — Phil Wang (MIT License)

---

### Related optimizer research

* Lion optimizer (momentum-free adaptive updates)
* GaLore-style gradient projection methods (SVD-based low-rank updates)
* Modular Muon implementations (Newton–Schulz optimization variants)

---

## License

This project inherits MIT-compatible licensing from its upstream components unless otherwise stated.

---

## Notes

This optimizer is experimental and combines multiple research ideas into a single unified update system. It may behave differently from standard AdamW in:

* convergence dynamics
* gradient scale behavior under Muon projection
* sensitivity to learning rate when RMS stabilization is active

Use with care in production settings.
