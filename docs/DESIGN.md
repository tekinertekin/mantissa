# Design notes

## 1. Numeric formats

All formats are IEEE-style `(-1)^sign · 1.mantissa · 2^(exp-bias)` with a
subnormal branch when the exponent field is zero and no-implicit-leading-one.
Only the bit budget differs.

| Format  | S | E | M | bias | max finite | min normal | mantissa digits |
|---------|:-:|:-:|:-:|:----:|-----------:|-----------:|:---------------:|
| float32 | 1 | 8 | 23 | 127 | 3.4e38 | 1.2e-38 | ~7.2 |
| fp16    | 1 | 5 | 10 | 15  | 65504  | 6.1e-5  | ~3.3 |
| bfloat16| 1 | 8 | 7  | 127 | 3.4e38 | 1.2e-38 | ~2.4 |
| tekin32 | 1 | 7 | 24 | 63  | 1.8e19 | 2.2e-19 | ~7.5 |
| tekin8  | 1 | 4 | 3  | 7   | 480    | 1.6e-2  | ~1.1 |
| e5m2    | 1 | 5 | 2  | 15  | 57344  | 6.1e-5  | ~0.9 |
| fp4     | 1 | 2 | 1  | 1   | 6      | 1.0     | ~0.3 |

References for the standard formats:
- **fp16 / mixed precision** — Micikevicius et al., *Mixed Precision Training*,
  ICLR 2018 (arXiv:1710.03740). IEEE 754-2008 binary16.
- **bfloat16** — Kalamkar et al., *A Study of BFLOAT16 for Deep Learning
  Training*, 2019 (arXiv:1905.12322). Same 8-bit exponent as float32, so it is
  literally the top 16 bits of a float32 — conversion is a shift.
- **fp8 E4M3 / E5M2** — Micikevicius et al., *FP8 Formats for Deep Learning*,
  2022 (arXiv:2209.05433), OCP FP8. E4M3 (tekin8) favors precision; E5M2 favors
  range and is used for gradients. tekin8 does not reserve the `S.1111.111` NaN
  slot, so its max finite is 480 rather than OCP's 448.
- **fp4 E2M1** — the element type of MXFP4/NVFP4 (OCP Microscaling v1.0, 2023;
  NVIDIA Blackwell). Eight magnitudes `{0,.5,1,1.5,2,3,4,6}`, no inf/nan; only
  usable inside a block-scaling scheme (roadmap), included here as the scalar
  primitive.

### Why bfloat16 beats fp16 for training

Both are 16 bits. fp16 spends them on precision (10 mantissa bits) with only 5
exponent bits, so its dynamic range is narrow (~6e-5 to 65504) and gradients
underflow without loss scaling. bfloat16 keeps float32's full 8-bit exponent
and spends only 7 bits on mantissa: far coarser, but the range never changes,
which is what makes it drop-in for training. The formats encode a genuine
range-vs-precision decision, not a quality ranking — that decision is the whole
subject of this library.

### tekin32 (1·7·24) — the design choice

The premise: a 32-bit float for neural networks does not need float32's ±3.4e38
range. Weights, activations, and loss-scaled gradients live comfortably inside
±1e18. Those unused exponent bits are pure waste.

tekin32 reallocates **one exponent bit into the mantissa**: 7 exponent bits
still cover ±1.8e19 (ample for any NN tensor), while 24 mantissa bits give a
25-bit significand — one bit *more* precise than float32. Consequences:

- Round-trips float32 **losslessly** for any value in `[2^-62, 2^63)`, i.e. the
  entire NN operating range (the 24th mantissa bit is zero-padded).
- Provides an extra bit of precision for values produced *inside* tekin32
  arithmetic or quantized down from float64 — useful as an accumulator or
  high-precision master-weight format.
- Costs range that NN workloads never use.

This is the bfloat16 philosophy inverted: bf16 assumes range is scarce and
sacrifices precision; tekin32 assumes range is abundant and buys precision back.
NVIDIA's TF32 (8 exp, 10 mantissa in a 32-bit lane) makes the opposite bet —
keep full range, cut precision to fp16 levels — which suits GPU matmul
throughput. tekin32 targets fidelity instead.

### tekin8 (1·4·3)

The 1-byte experiment: 4× smaller than float32. With 3 mantissa bits its
relative step is ~1/8, so it is only usable where quantization noise is
tolerable (post-training weight quantization, activations feeding a robust
layer). Included to make the memory-vs-accuracy trade visible at its extreme —
running the test suite across formats shows exactly where 3 mantissa bits stop
being enough.

## 2. Performance

### Store narrow, accumulate wide

Inputs are read as narrow types; every dot-product accumulator is float32
(`tk_dot`, `ops.c`). This is the mixed-precision recipe (Micikevicius 2017):
narrow storage cuts memory traffic, float32 accumulation keeps the error of a
sum over millions of terms bounded. Accumulating in the narrow type instead
would lose low-order bits catastrophically for large layers.

### Memory bandwidth is the budget

A dense layer's forward pass (GEMV) reads the whole weight matrix once and does
~2 FLOPs per weight — when the matrix exceeds cache it is **memory-bandwidth
bound**, so halving the storage size approaches halving the runtime. `make
bench` confirms the direction: bfloat16 (half the bytes, a one-shift read) edges
out float32.

But the benchmark also tells the honest counter-story: **tekin8 is slower**, not
4× faster, once the matrix is cache-resident — its E4M3→float conversion (a
subnormal branch, not SIMD-friendly) now dominates the arithmetic. The lesson is
real: narrow storage only converts to speed when the read is cheap (a shift) or
hardware-accelerated (F16C, AVX512-BF16, Blackwell FP8 tensor cores). This is
why the conversion hot path matters as much as the byte count.

### Conversion: hot path vs cold path

- **narrow → float32 (hot)**: runs for every weight, every forward pass. These
  are branchless `static inline` bit manipulations in `dtypes.h` so they inline
  into `tk_dot` and vectorize. bf16 is a single shift.
- **float32 → narrow (cold)**: runs once, when weights are quantized/loaded, so
  `dtypes.c` favors clear, correct code (`frexpf`/`lroundf`).

A production hot path would replace the scalar reads with hardware conversion —
`_mm256_cvtph_ps` (F16C) for fp16, AVX512-BF16 for bfloat16. The split keeps
that door open without complicating the cold path.

### The dot-product loop

`tk_dot` uses four independent accumulators. A single accumulator serializes on
the floating-point adder's latency (a loop-carried dependency); four let the
out-of-order engine keep multiple FMAs in flight. Combined with
`-ffp-contract=fast` (fold `a*b + c` into one FMA), `restrict` (no
aliasing → free to vectorize), and `-O3 -funroll-loops`, the compiler emits
vectorized FMA. `-march=native` unlocks the full SIMD width locally.

### Activation dispatch: switch beats a function pointer

Intuition says an indirect `function pointer` avoids the cost of a `switch`.
The benchmark says otherwise: applying an activation element-wise, the inline
`switch` runs ~3× faster than a resolved pointer for `relu`. Reason — the
`switch` body inlines and the loop **vectorizes**; an indirect call per element
is opaque to the vectorizer and pays call overhead each iteration. So
`tk_activate` uses the `switch`, and `tk_act_resolve()` (the pointer table) is
kept only for genuinely pluggable dispatch (e.g. a caller-supplied activation).
A dispatch mechanism is only "faster" if you measure it in the loop it runs in.

## 3. Reuse across architectures

`tk_linear_forward` — `activation(W·x + bias)` with an optional (NULL-able)
bias — is deliberately the only "layer". It is the shared primitive of:

- **MLP / ANN**: stack it.
- **RNN / GRU / LSTM**: apply it per timestep; gates are just linear projections
  followed by sigmoid/tanh, both provided.
- **Transformer**: Q/K/V and feed-forward are linear projections; attention
  output projections are typically bias-free (hence the NULL bias path), and
  GELU (Hendrycks & Gimpel, 2016, arXiv:1606.08415) is provided.
- **CNN**: a convolution is a batch of dot products of a filter over input
  patches. `tk_dot` is that inner product already; a conv layer adds an
  `im2col`/patch iterator (or a direct sliding window) on top and reuses the
  same narrow-store/float-accumulate numerics — no new arithmetic. Planned.

Heterogeneous models are supported today: bias is a per-call NULL-able pointer
and the activation is a per-call argument, so each layer chooses independently
(see `examples/mlp_example.c`).

Building this one primitive well now means those models reuse it later instead
of reimplementing the numerics.

## 4. Numeric landscape (context for the format choices)

- **Microscaling (MX)** — OCP MX v1.0 (2023): a block of 32 elements shares one
  E8M0 scale, restoring the dynamic range that 4/6-bit elements lack. Element
  types MXFP8 (E4M3/E5M2), MXFP6, MXFP4 (E2M1). mantissa implements the element
  formats; a shared per-block scale is the next step.
- **NVFP4** — NVIDIA Blackwell (2024): 16-element blocks, an FP8 E4M3 block
  scale plus a per-tensor FP32 scalar; LLMs pretrained at 4 bits
  (arXiv:2509.25149).
- **Posit / takum** (Gustafson) — tapered-precision alternatives to IEEE
  floats; more precision near ±1 where zero-centered weights concentrate.
- **IEEE P3109** — emerging standard for ML arithmetic formats (2025).

## 5. Back-propagation

The backward pass keeps the forward core's philosophy: no autograd graph, an
explicit primitive the caller drives.

`tk_linear_backward` collapses the activation derivative into the output
gradient once (`dz = dy · act'(z)`), then computes `dW`, `db`, and `dx` in a
single pass over each weight row — `W`/`dW` accessed sequentially (cache
friendly) while `dx` stays hot across the output loop. Accumulation is float32,
matching the forward pass.

**Stochastic rounding** is the key low-precision-training idea (Gupta et al.,
2015, arXiv:1502.02551). Storing weights narrow, a plain round-to-nearest write
of `w - lr·g` discards any update below the type's ULP, so with a small learning
rate training silently stalls. SR rounds to the neighbouring grid point with
probability equal to the fractional distance, so an update of ⅓ ULP moves the
weight ⅓ of the time — correct in expectation. `tk_sr_from_float` derives the
ULP from the value's binade and `TK_MANT_BITS`, rounds onto the grid, and the
final `TK_FROM_FLOAT` is then exact. This is what lets `make train` learn XOR in
bfloat16 without an fp32 master weight copy; on float32/tekin32 it is a no-op.
It is the same mechanism used for FP8 weight updates on Hopper/Blackwell.

**L1/L2** fold into the gradient at the optimizer step (`g += l2·w`,
`g += l1·sign(w)`); **dropout** is inverted (survivors scaled by `1/(1-rate)`),
with the mask reused on the backward pass. All are config-gated, off by default.

**Correctness** is not asserted, it is checked: `make testbp` compares analytic
gradients against central finite differences (`(L(w+h)−L(w−h))/2h`) at float32,
requiring <1e-2 relative error over all weight and input gradients for tanh,
sigmoid, relu, and gelu. Finite-difference gradient checking is the standard way
to prove a hand-written backward pass matches its forward pass.
