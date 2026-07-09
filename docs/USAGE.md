# Usage

## Build

Zero config gives you bfloat16 + bias. Override the storage format with `DTYPE`
(see `config.h`):

```sh
make test            # tests (default bfloat16)
# DTYPE: 0 float32  1 fp16  2 bfloat16  3 tekin32  4 tekin8  5 fp8_e5m2  6 fp4_e2m1
make DTYPE=4 test    # switch to tekin8
make example         # C perceptron   (build/perceptron_example)
make mlp             # mixed 3-layer MLP
make DTYPE=0 bench   # GEMV + activation-dispatch benchmark
make dist            # shared library for Python (dist/libmantissa.{so,dylib,dll})
```

`TK_USE_BIAS=0` flips the default-bias policy the examples follow (the core API
controls bias per-call regardless — pass `NULL`).

## C API

### Types (`dtypes.h`)

`tk_scalar_t` is the active storage type. Move values in and out of it with the
config-selected macros — your code stays dtype-agnostic:

```c
tk_scalar_t w = TK_FROM_FLOAT(0.5f);   // float -> storage
float       f = TK_TO_FLOAT(w);        // storage -> float
```

Explicit converters (all exported for the binding): `tk_float_to_fp16` /
`tk_fp16_to_float`, and the `bf16` / `t32` / `f8` / `e5m2` / `fp4` equivalents.

### Activations (`activations.h`)

`tk_activation_t`: `TK_ACT_IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU`.
`tk_activate(float *y, int n, act)` applies one over a vector in place (inline
switch — vectorizes). `tk_act_resolve(act)` returns a `tk_act_fn` function
pointer for pluggable dispatch; prefer `tk_activate` for the built-ins (faster,
see the benchmark).

### Linear layer (`ops.h`)

```c
float tk_dot(const tk_scalar_t *a, const tk_scalar_t *b, int n);

void  tk_linear_forward(const tk_scalar_t *W,     // out_dim x in_dim, row-major
                        const tk_scalar_t *x,     // in_dim
                        const tk_scalar_t *bias,  // out_dim, or NULL
                        float *y,                 // out_dim (float32 out)
                        int out_dim, int in_dim,
                        tk_activation_t act);
```

### Example: a perceptron in C

A 2-input OR gate is one neuron: `y = step(w·x + b)`.

```c
#include "ops.h"

enum { IN = 2, OUT = 1 };
tk_scalar_t W[OUT * IN] = { TK_FROM_FLOAT(1.0f), TK_FROM_FLOAT(1.0f) };
tk_scalar_t b[OUT]      = { TK_FROM_FLOAT(-0.5f) };

tk_scalar_t x[IN] = { TK_FROM_FLOAT(1.0f), TK_FROM_FLOAT(0.0f) };
float y;
tk_linear_forward(W, x, b, &y, OUT, IN, TK_ACT_STEP);   // y == 1
```

Full file: [`examples/perceptron_example.c`](../examples/perceptron_example.c)
(`make example`).

## Python API

`make dist` (any dtype — the binding is dtype-agnostic), then:

```python
from mantissa import Mantissa, STEP

tk = Mantissa()                     # loads ../dist/libmantissa.<ext>
print(tk.dtype)                     # e.g. 'bfloat16'
W    = [1.0, 1.0]                   # 1 x 2, row-major
bias = [-0.5]
y = tk.linear_forward(W, [1, 0], bias, out_dim=1, in_dim=2, act=STEP)
print(int(y[0]))                    # 1
```

Values are quantized through the library's storage dtype inside
`tk_linear_forward_f32`, so the same Python runs against any build.
Full file: [`python/perceptron_example.py`](../python/perceptron_example.py).

## Which functions to call

| You want to… | Call |
|---|---|
| Quantize a float weight into storage | `TK_FROM_FLOAT` / `tk_float_to_*` |
| Read a stored weight back as float | `TK_TO_FLOAT` / `tk_*_to_float` |
| One neuron / a dense layer forward | `tk_linear_forward` |
| Just a dot product | `tk_dot` |
| Apply an activation over a vector | `tk_activate` |
| Know the active dtype from Python | `Mantissa().dtype` |
