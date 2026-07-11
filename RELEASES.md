# Release notes

Version history for **mantissa**, newest first. Updated on every release;
benchmark numbers are from an Apple M-series laptop (clang `-O3`), a 2048×2048
dense layer (4.2M params) unless noted, and are indicative, not absolute.

---

## v0.1.7 — 2026-07-10  (tag `v0.1.7`)

Code-review pass over the hot paths. Each suggestion was benchmarked before
being accepted; only the change that actually helped was kept.

**Optimized**
- `tk_linear_forward_f32` (the float32-facing binding entry point) now quantizes
  the input vector `x` **once** into a scratch buffer instead of re-quantizing it
  on every output row, removing the x-side round-trip from the inner loop
  (~half the per-element conversions in that path). Output is unchanged; the
  narrow-storage hot path `tk_linear_forward` never had this cost.

**Reviewed and *not* changed** (kept for the record, since "measure, don't
assume" cuts both ways):
- *Explicit NEON for the backward pass* — benchmarked and **rejected**: it
  matched the auto-vectorized `dW` path and was ~40% *slower* for the `dx`
  reduction, so the compiler's code was already better.
- *Thread pool overhead on tiny layers* — already handled: `TK_MT_MIN_WORK`
  keeps small layers fully serial (they never touch the pool). A spin-wait was
  declined: it would burn cores when idle, wrong for a general-purpose library.
- *"Cache thrash" / vectorization stall in the backward `dx += …`* — the inner
  loop is element-wise (not a reduction over `i`), and `dx` fits L1 for realistic
  `in_dim`, so the premise did not hold.
- *Branchless dropout mask / L1 sign* — `-O3` already lowers these data-dependent
  selects to branchless `csel`/`cmov`, so a manual rewrite changes nothing.

All 7 dtypes, the gradient check, and the examples pass, warning-free.

---

## v0.1.6 — 2026-07-10  (tag `v0.1.6`)

Multithread the **backward** pass too (v0.1.5 did the forward).

**Optimized**
- `tk_linear_backward` splits output rows across the thread pool. `dW`/`db` are
  per-row so they stay bitwise-identical to serial; `dx` is a reduction over
  rows, so when it is requested each worker accumulates into a private buffer
  and the partials are summed afterwards (dx then differs only by float
  reduction order). When `dx == NULL` (e.g. the first layer) the whole pass is
  bitwise-identical.

**Benchmarks** (bfloat16 backward GEMV, 2048×2048, 10-core Apple laptop)
| threads | ms/pass | GFLOP/s |
|---|:--:|:--:|
| 1  | 1.01 | 12.45 |
| 10 | 0.33 | **38.67** (3.1×) |

**Correctness:** verified `dW`/`db` bitwise-identical serial vs 10 threads, and
`dx` matching to reduction-order tolerance; all tests + gradient check pass.

---

## v0.1.5 — 2026-07-10  (tag `v0.1.5`)

Multithreaded GEMV — split the dense-layer output rows across CPU cores.

**Added**
- Persistent fork-join **thread pool** (`src/pool.c`): workers created once
  (lazily), woken per call by a condvar barrier — no per-call `pthread_create`.
  `tk_linear_forward` splits output rows across it above a work threshold; small
  layers stay serial so the pool never hurts the millions-of-small-calls path.
  `MANTISSA_THREADS` tunes the count; non-pthreads targets get a serial stub.

**Benchmarks** (bfloat16 forward GEMV, 2048×2048, 10-core Apple laptop)
| threads | GFLOP/s |
|---|:--:|
| 1  | ~29 |
| 10 | **~83** (2.9×) |

Sub-linear by design: GEMV does ~2 FLOPs/byte, so it hits the memory-bandwidth
wall before the compute wall (float32, twice the bytes, gains only ~1.2×).
**Correctness:** the multithreaded result is bitwise identical to serial (each
row is computed entirely by one thread — no reduction reordering); verified by
matching checksums and the full test suite.

---

## v0.1.4 — 2026-07-10  (tag `v0.1.4`)

NEON kernel for **bfloat16** (the default dtype) — the follow-up promised in
v0.1.3.

**Optimized**
- Explicit NEON GEMV for bfloat16 on arm64: 4 values loaded with `vld1_u16` and
  widened in-register via `vshll_n_u16(v, 16)` (that shift *is* the bf16→float32
  conversion), then FMA'd. Portable fallback unchanged elsewhere.

**Benchmarks** (forward GEMV, 2048×2048; indicative, laptop-noisy)
| dtype | baseline | v0.1.3 | v0.1.4 |
|---|:--:|:--:|:--:|
| bfloat16 | 8.57 | 10.70 | **~21** GFLOP/s |
| float32  | 7.97 | 14.29 | ~15 |
| tekin8   | 2.55 | 2.76  | ~3.2 |

bfloat16 is now the fastest dtype *and* the smallest: it moves half the bytes of
float32 for the same FMA work. All 7 forward dtypes + gradient check pass; the
bf16 NEON path is validated by the bf16 tests and the XOR trainer on arm64.

---

## v0.1.3 — 2026-07-10  (tag `v0.1.3`)

Matmul (GEMV) throughput — the actual bottleneck in a real network.

**Optimized**
- **Register-blocked forward pass** (`tk__dot4`): computes four output rows at
  once so each `x` element is loaded/converted once and feeds four independent
  FMA chains. ~1.3× across all dtypes, portable.
- **Explicit NEON kernel** for the float32 build on arm64 (`vfmaq_f32` +
  `vaddvq_f32`), with the portable path as the fallback on every other
  target/dtype. ~1.8× for float32.

**Benchmarks** (forward GEMV, 2048×2048)
| dtype | v0.1.2 GFLOP/s | v0.1.3 GFLOP/s | speedup |
|---|:--:|:--:|:--:|
| float32  | 7.97 | 14.29 | 1.8× (NEON) |
| bfloat16 | 8.57 | 10.70 | 1.25× |
| tekin8   | 2.55 | 2.76  | 1.08× |

Correctness re-verified: all 7 forward dtypes + the gradient check pass. The
NEON path is exercised by the float32 tests on arm64 (locally); CI runs on
x86_64 and covers the portable fallback. Next: a bf16-widening NEON kernel and
an AVX path.

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
