# Release notes

Version history for **mantissa**, newest first. Updated on every release;
benchmark numbers are from an Apple M-series laptop (clang `-O3`), a 2048×2048
dense layer (4.2M params) unless noted, and are indicative, not absolute.

---

## v0.1.2 — 2026-07-10  (tag `v0.1.2`)

Multi-language clients, Python training, README diagrams, and an
externally-reviewed optimization pass.

**Added**
- **Python back-prop**: `tk_train_step_f32` (one float32 call = forward + MSE +
  backward + SGD update for a dense layer), wrapped as `Mantissa.train_step`;
  `python/train_example.py`.
- **`clients/`** — forward + back-prop demos calling the downloaded library:
  C++ (`dlopen`), C# (P/Invoke), Java (JNA), JavaScript (Node/koffi), Rust
  (`libloading`). C++, Node, and Python verified locally.
- **README diagrams** (Mermaid): colorized call-flow architecture + a
  training-loop figure.

**Optimized** (reviewed suggestions kept only where safe / measured)
- Stochastic-rounding ULP computed by exponent bit arithmetic instead of
  `frexpf`/`ldexpf` — two fewer libm calls per weight, numerically identical.
- RNG state hoisted into a register across the SGD / dropout loops.
- GELU gradient reordered to reuse `z²` (FMA-friendly).
- Opt-in **`TK_FAST_MATH`** (default off): branchless rational `tanh`/`sigmoid`
  that vectorizes instead of calling libm.

**Benchmarks**
| pass (bf16) | ms/pass | GFLOP/s |
|---|:--:|:--:|
| forward  | 0.98 | 8.6 |
| backward | 0.98 | 12.8 |

| SGD weight update | M weights/s |
|---|:--:|
| float32 | 9379 |
| bfloat16 | 994 |
| tekin8 (stochastic rounding) | 321  *(was 298 in v0.1.1)* |

| sigmoid over 4M values | ms |
|---|:--:|
| exact (libm, default) | 4.42 |
| `TK_FAST_MATH=1` | 1.15  *(~3.8× faster, ~2% error)* |

Honest note: these micro-optimizations are modest because `-O3` already does
much of it, and activations are O(n) vs the layer's O(n²) matmul — so `TK_FAST_MATH`
only helps activation-bound workloads. The matmul (GEMV) remains the lever.

---

## v0.1.1 — 2026-07-10  (commit `c61231a`)

Performance framing, branchless activations, CI, and downloadable binaries.

**Added**
- **CI** (`ci.yml`): all 7 storage dtypes + the back-prop gradient check on
  every push; live badge in the README.
- **Release workflow** (`release.yml`): every `v*` tag builds the shared library
  on Linux/macOS/Windows and attaches them to the GitHub Release — download and
  use with no toolchain.
- Back-propagation benchmark (`bench/bench_backprop.c`).

**Changed**
- README rebuilt to lead with speed/memory; storage formats demoted to "the
  precision dial"; bit-level detail moved to `docs/DESIGN.md`.
- Branchless activation kernels, each picked by benchmark: `step` via sign-bit
  (~40% faster than compare), `relu` via `fmax`, `sign` via `copysign`.
- Prebuilt library removed from git (now shipped via releases).

**Benchmarks**
| GEMV forward | weight mem | ms/pass | GFLOP/s |
|---|:--:|:--:|:--:|
| float32  | 16 MB | 1.05 | 7.97 |
| bfloat16 |  8 MB | 0.98 | 8.57 |
| tekin8   |  4 MB | 3.29 | 2.55 |

| `relu` over 4M values | ms |
|---|:--:|
| before (ternary) | 1.16 |
| after (`fmax`, branchless) | 0.41 |

Stochastic rounding vs round-to-nearest — XOR final loss after 4000 epochs:

| dtype | round-to-nearest | stochastic |
|---|:--:|:--:|
| float32  | 0.00008 | 0.00008 |
| bfloat16 | 0.01090 | 0.00009 |
| tekin8   | 0.24862 *(stalled)* | 0.00008 *(converged)* |

---

## v0.1.0 — 2026-07-10  (commit `73bee96`)

First tagged release: the numeric core plus training.

**Added**
- Seven storage formats: `float32`, `fp16`, `bfloat16`, custom `tekin32`
  (1·7·24), `tekin8` (FP8 E4M3), `fp8_e5m2`, `fp4_e2m1` — selected via `DTYPE`,
  default `bfloat16`. Narrow storage, float32 accumulation.
- Dense-layer forward (`tk_linear_forward`, GEMV), activations
  (step/sign/relu/sigmoid/tanh/gelu), mixed per-layer bias & activation.
- **Back-propagation**: `tk_linear_backward`, `tk_loss` (MSE/BCE), `tk_sgd_step`,
  dropout, L1/L2, and **stochastic rounding** — all training options off by
  default. Validated by a finite-difference gradient check (`make testbp`) and
  by learning XOR in bfloat16 (`make train`).
- ctypes Python binding, benchmark, MIT license.
