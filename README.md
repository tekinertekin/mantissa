# mantissa

A small, fast, low-precision numerics core for neural networks, written in C
with a Python (ctypes) binding. The C layer stores weights and activations in
configurable narrow float formats and computes dense-layer forward passes; the
Python layer is the user-facing surface.

The design goal is the one that matters at scale: **shrink memory and stay
fast**. A parameter stored in 1 byte instead of 4 is a 4× cut in the memory
(and VRAM) a model occupies, and a matrix-vector product bound by memory
bandwidth speeds up by roughly the same factor. Every choice here serves that.

> Started by Tekin Ertekin (2024); later refactored with Claude Code — see
> [AUTHORS.md](AUTHORS.md).

## Numeric formats

Weights and activations are stored in one selectable format but **accumulated in
float32** — the standard mixed-precision recipe (narrow store, wide accumulate;
Micikevicius et al., 2017).

| `DTYPE` | Name       | Layout (S·E·M) | Bytes | Notes |
|--------:|------------|----------------|:-----:|-------|
| 0 | `float32`  | 1·8·23 | 4 | IEEE-754 binary32, the reference |
| 1 | `fp16`     | 1·5·10 | 2 | IEEE-754 half (Micikevicius 2017) |
| 2 | `bfloat16` | 1·8·7  | 2 | Google bfloat16 (Kalamkar 2019) |
| 3 | `tekin32`  | 1·7·24 | 4 | custom: NN range traded for extra precision |
| 4 | `tekin8`   | 1·4·3  | 1 | custom FP8 E4M3 (Micikevicius 2022) |

`tekin32` and `tekin8` are original designs; the reasoning (why these exact
bit splits) is in [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
include/    config, data types + conversions, activations, linear-algebra ops
src/        implementations
tests/      round-trip + forward-pass checks for all 5 formats
examples/   C perceptron built from the primitives
python/     ctypes binding + Python perceptron
docs/       DESIGN.md (numerics + optimization), USAGE.md (API + examples)
```

## Quick start

```sh
make DTYPE=2 test     # build + run tests with bfloat16 storage
make example          # C perceptron
make lib              # shared library for Python
python3 python/perceptron_example.py
```

Full API and worked C/Python examples: [docs/USAGE.md](docs/USAGE.md).

## Scope

This is the numeric core. It is deliberately general: a dense layer
(`W·x + bias`, then an activation) is the shared building block of MLPs, RNNs,
GRUs, LSTMs, and Transformers, so the same primitives will back those models as
the project grows. Training / back-propagation is a later stage and is not here
yet.
