# mantissa

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Language](https://img.shields.io/badge/C-C11-00599C.svg)
![Python](https://img.shields.io/badge/python-ctypes-3776AB.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-7%20dtypes%20%2B%20gradcheck-brightgreen.svg)

**A low-precision numerics core for neural networks — C engine, Python skin.**

`mantissa` stores weights and activations in configurable narrow floating-point
formats and runs dense-layer forward passes on them. It exists to make one
trade-off explicit and measurable: **how few bits per parameter can you spend
before accuracy breaks — and what does that buy in memory and speed?** That
question decides whether a model fits in VRAM and how fast it runs.

> Started by Tekin Ertekin (2024); later refactored with Claude Code — see
> [AUTHORS.md](AUTHORS.md).

---

## Why this exists

A parameter stored in 1 byte instead of 4 is a **4× cut** in the RAM/VRAM a
model needs. And because a dense layer's forward pass is bound by memory
bandwidth, less data moved is also less time. The entire library is organized
around the storage format because that is the real lever at scale.

The recipe throughout is **narrow storage, float32 accumulation** — the
standard mixed-precision approach (Micikevicius et al., 2017): store small,
sum wide, so error stays bounded across the millions of terms in a layer.

## Numeric formats

| `DTYPE` | Name       | Layout (S·E·M) | Bits | Source |
|--------:|------------|:--------------:|:----:|--------|
| 0 | `float32`  | 1·8·23 | 32 | IEEE-754 binary32 (reference) |
| 1 | `fp16`     | 1·5·10 | 16 | IEEE-754 half (Micikevicius 2017) |
| 2 | `bfloat16` | 1·8·7  | 16 | Google bfloat16 (Kalamkar 2019) — **default** |
| 3 | `tekin32`  | 1·7·24 | 32 | **custom** — range traded for precision |
| 4 | `tekin8`   | 1·4·3  | 8  | FP8 E4M3 (Micikevicius 2022) |
| 5 | `fp8_e5m2` | 1·5·2  | 8  | FP8 E5M2, range variant (OCP / IEEE P3109) |
| 6 | `fp4_e2m1` | 1·2·1  | 4  | FP4 E2M1, the MXFP4/NVFP4 element (OCP MX 2023) |

Every value is stored in the selected type but computed in float32. Conversions
split into a branchless **hot path** (narrow→float, inlined into the dot loop)
and a clear **cold path** (float→narrow, run once at weight load).

### tekin32 — why a *second* 32-bit float?

`float32` is `1·8·23`. `tekin32` is `1·7·24` — same 32 bits, one exponent bit
moved into the mantissa. The reasoning:

- float32's 8-bit exponent spans **±3.4e38**. Neural-network weights,
  activations, and even loss-scaled gradients live far inside **±1e18** — that
  range is mostly unused.
- tekin32 spends 7 exponent bits (still **±1.8e19**, ample) and gives the freed
  bit to the mantissa: a **25-bit significand vs float32's 24**.

So `tekin32`:
- **round-trips float32 losslessly** across the whole NN operating range, and
- carries **one extra bit of precision** for values produced inside tekin32
  arithmetic or quantized down from float64 (a high-fidelity accumulator).

It is the bfloat16 philosophy *inverted*: bf16 assumes range is scarce and
sacrifices precision; tekin32 assumes range is abundant and buys precision back.
Full derivation in [docs/DESIGN.md](docs/DESIGN.md).

## Precision, seen

The same value stored in each format and read back (`make DTYPE=<n> test`).
Precision tracks the mantissa budget; `float32` is the reference column:

| input   | float32 | tekin32 | fp16      | bfloat16  | e5m2 | tekin8 |
|---------|--------:|--------:|----------:|----------:|-----:|-------:|
| 3.14159 | 3.14159 | 3.14159 | 3.14062   | 3.14062   | 3.0  | 3.25   |
| 100.0   | 100     | 100     | 100       | 100       | 96   | 104    |
| 0.01    | 0.01    | 0.01    | 0.0100021 | 0.0100098 | 0.0098 | 0.00977 |

`fp4_e2m1` is coarser still: its 8 magnitudes are `{0, ±0.5, ±1, ±1.5, ±2, ±3,
±4, ±6}` — everything rounds onto that grid.

## Benchmark

`make DTYPE=<n> bench` — a 2048×2048 dense layer (4.2M params), plus the
activation-dispatch micro-test. Numbers below are from an Apple M-series laptop
(clang `-O3`); they are indicative, not absolute.

| dtype    | weight memory | GEMV ms/pass | GEMV GFLOP/s |
|----------|:-------------:|:------------:|:------------:|
| float32  | 16.0 MB | 1.05 | 7.97 |
| bfloat16 |  8.0 MB | 0.98 | 8.57 |
| tekin8   |  4.0 MB | 3.29 | 2.55 |

Reading, honestly: `bfloat16` beats `float32` (half the bytes, bf16→float is a
single shift). `tekin8` is *slower* despite ¼ the memory — its E4M3→float
conversion (a subnormal branch, no SIMD) dominates once the matrix fits in
cache. On hardware with native FP8 conversion (F16C, AVX512-BF16, Blackwell
tensor cores) that cost disappears; here it is a candid picture of a scalar,
portable implementation.

**Activation dispatch** (4M elements): a per-element `switch` beats a resolved
function pointer ~3× for `relu` and ~1.5× for `sigmoid`. The inline `switch`
vectorizes; an indirect call per element does not. So `tk_activate` keeps the
`switch`, and the function-pointer API (`tk_act_resolve`) is reserved for
genuinely pluggable dispatch. *Measure, don't assume.*

## Mixed architectures

Every layer configures itself independently — bias is a per-call NULL-able
pointer, activation is a per-call argument:

```c
tk_linear_forward(W1, x,  b1,   h1, 6, 4, TK_ACT_TANH);     /* bias + tanh    */
tk_linear_forward(W2, h1, NULL, h2, 5, 6, TK_ACT_RELU);     /* no bias + relu */
tk_linear_forward(W3, h2, b3,   y,  2, 5, TK_ACT_SIGMOID);  /* bias + sigmoid */
```

That is a full 3-layer MLP with three different bias/activation setups —
exactly the heterogeneity a Transformer needs (bias-free attention projections,
bias-using feed-forward). Run it: `make mlp`.

## Training (back-propagation)

The reverse pass mirrors the forward core: `tk_linear_backward` computes the
weight, bias, and input gradients for a dense layer; `tk_sgd_step` updates
narrow-stored weights; `tk_loss` (MSE / BCE) seeds the gradient. No autograd
graph — the caller drives the layers, keeping everything explicit and
inspectable.

Correctness is proven by a **gradient check** (`make testbp`): analytic
gradients vs central finite differences, matching to <1e-3 for tanh, sigmoid,
relu, and gelu.

`make train` learns **XOR** — the problem a single perceptron *cannot* solve —
end to end, in the default **bfloat16**:

```
Training XOR  (dtype=bfloat16, stochastic_rounding=1)
  epoch    0  loss 0.31523
  epoch 1000  loss 0.00035
  epoch 4000  loss 0.00007
predictions:
  (0,0) -> 0.004   (0,1) -> 0.990   (1,0) -> 0.991   (1,1) -> 0.009
```

That it converges in bf16 is the point of **stochastic rounding** (config
`TK_USE_STOCHASTIC_ROUNDING`): under plain round-to-nearest, a weight update
smaller than the storage type's precision rounds to zero and training stalls; SR
rounds up/down with probability proportional to distance, so tiny updates
accumulate in expectation (Gupta et al., 2015; the technique behind FP8 training
on Hopper/Blackwell). It needs no fp32 master copy of the weights.

### Training config (all OFF by default)

| Flag | Effect |
|------|--------|
| `TK_USE_DROPOUT` / `TK_DROPOUT_RATE` | inverted dropout on activations |
| `TK_USE_L1` / `TK_L1_LAMBDA`         | L1 weight penalty in the update |
| `TK_USE_L2` / `TK_L2_LAMBDA`         | L2 weight penalty in the update |
| `TK_USE_STOCHASTIC_ROUNDING`         | SR on the weight write-back |

Each sets a default; the runtime `tk_optim` / dropout calls override per layer.
Back-propagation is scoped to the dense layer for now (the shared primitive);
conv/recurrent backward reuse the same pieces later.

## Zero-config

Never touch `config.h` and you get Google's **bfloat16** with **bias enabled** —
the safe general-purpose default. Override only when you want to:

```sh
make DTYPE=4 test     # switch storage to tekin8
```

## Quick start

```sh
make test                     # tests, default bfloat16
make example                  # C perceptron
make mlp                      # mixed 3-layer MLP
make DTYPE=0 bench            # benchmark, float32
make dist && python3 python/perceptron_example.py
```

The Python binding is dtype-agnostic: it calls the float32 entry point, so the
same script runs against any storage type the library was built for. Full API
and examples in [docs/USAGE.md](docs/USAGE.md).

## Numeric landscape & roadmap

The low-precision frontier moves fast; `mantissa` tracks it deliberately:

- **Microscaling (MX)** — OCP MX v1.0 (2023): blocks of 32 elements share one
  E8M0 scale, mitigating the tiny dynamic range of 4/6-bit elements. `MXFP8`,
  `MXFP6`, `MXFP4`.
- **NVFP4** — NVIDIA Blackwell (2024): 16-element blocks with an FP8 E4M3 block
  scale; used to pretrain LLMs at 4 bits (arXiv:2509.25149).
- **Posit / takum** — tapered-precision alternatives to IEEE floats, strong for
  ≤8-bit inference on zero-centered weights.
- **IEEE P3109** — an emerging standard for ML arithmetic formats.

`mantissa` implements the *element* formats (E4M3, E5M2, E2M1); block-level
microscaling (a shared per-block scale) is the next planned step. **Convolution
is already within reach**: a conv layer is a batch of dot products of a filter
against input patches, and `tk_dot` is exactly that primitive — a CNN needs an
`im2col`/patch iterator on top, no new numerics. Back-propagation for the dense
layer is implemented (gradient-checked); conv/recurrent backward will reuse the
same gradient and optimizer pieces.

## Project layout

```
include/   config, dtypes + conversions, activations, ops, loss, backprop
src/       implementations
tests/     forward round-trip checks (7 formats) + backprop gradient check
examples/  perceptron, mixed-MLP, XOR training
bench/     GEMV + activation-dispatch benchmark
python/    ctypes binding + Python perceptron
docs/      DESIGN.md (numerics, optimization), USAGE.md (API + examples)
```

## License

MIT — see [LICENSE](LICENSE). © 2024 Tekin Ertekin.
