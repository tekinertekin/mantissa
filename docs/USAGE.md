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

// Narrow n floats into the storage type. For repeated inference from float
// buffers: quantize W once with this, then call tk_linear_forward — instead of
// paying tk_linear_forward_f32's per-call re-quantization of every weight
// (~9x serial / ~26x threaded on a 2048x2048 bf16 layer).
void  tk_quantize(const float *src, tk_scalar_t *dst, int n);
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

Install from PyPI — `pip install mantissa-nn` (import name stays `mantissa`; the
wheel bundles a prebuilt library, no toolchain needed; add `[numpy]` for the
zero-copy path). Or, from a checkout, `pip install .` (compiles the C core) or
just `make dist` (any dtype — the binding is dtype-agnostic). Then:

```python
from mantissa import Mantissa, IDENTITY, STEP

tk = Mantissa()                     # loads ../dist/libmantissa.<ext>
print(tk.dtype)                     # e.g. 'bfloat16'
W    = [1.0, 1.0]                   # 1 x 2, row-major
bias = [-0.5]
y = tk.linear_forward(W, [1, 0], bias, out_dim=1, in_dim=2, act=STEP)
print(int(y[0]))                    # 1
```

Training from Python is one call per step (`tk.train_step`, wrapping
`tk_train_step_f32` — forward + gradient + SGD update on float32 arrays):

```python
W, bias = [0.0, 0.0, 0.0], [0.0]
for _ in range(200):
    loss = tk.train_step(W, [1, 2, 3], [14.0], out_dim=1, in_dim=3,
                         act=IDENTITY, lr=0.01, bias=bias)   # W, bias updated in place
```

Plain lists work everywhere, but they are boxed element-by-element on every
call. For real training loops pass **float32 numpy arrays or `array('f')`**
instead — the binding then hands the buffer to C zero-copy and mutates it in
place (~200x faster per step at 512x512; numpy is optional, nothing breaks
without it):

```python
import numpy as np
W    = np.zeros(3,  dtype=np.float32)
bias = np.zeros(1,  dtype=np.float32)
x    = np.array([1, 2, 3], dtype=np.float32)
t    = np.array([14.0],    dtype=np.float32)
for _ in range(200):
    loss = tk.train_step(W, x, t, out_dim=1, in_dim=3,
                         act=IDENTITY, lr=0.01, bias=bias)  # zero-copy, in place
```

When training over a *dataset*, don't cross the FFI once per sample — pass the
whole epoch (`X`: `n_samples x in_dim` rows, flat) and let C drive the loop.
Same sequential SGD, bit-identical weights, one crossing per epoch
(**~140x faster** at 1000x4 than per-sample `train_step` calls):

```python
loss = tk.train_epoch(W, X, targets, n_samples=1000, out_dim=1, in_dim=4,
                      act=IDENTITY, lr=0.01, bias=bias)   # mean loss
```

For repeated *inference*, pass `out=` to `linear_forward` (a float32 numpy
array or `array('f')` of `out_dim`): the result is written straight into your
buffer — no per-call output allocation or boxing (~1.24x per call at 256x256).

Full files: [`python/perceptron_example.py`](../python/perceptron_example.py),
[`python/train_example.py`](../python/train_example.py). The same two functions
back the other-language demos in [`clients/`](../clients).

## Training (back-propagation)

```c
#include "loss.h"
#include "backprop.h"

float loss = tk_loss(y, target, dy, n, TK_LOSS_MSE);   // seeds dL/dy

// dense-layer backward: fills dW, db (or NULL), dx (or NULL)
tk_linear_backward(W, x, z /*pre-activation*/, dy, dW, db, dx,
                   out_dim, in_dim, act);

tk_optim opt = tk_optim_default(0.5f);   // lr; l1/l2/SR from config
tk_sgd_step(W, dW, out_dim * in_dim, &opt, &rng);   // W -= lr*(dW + reg)
```

- `z` is the pre-activation — get it with `tk_linear_forward(..., TK_ACT_IDENTITY)`
  then `tk_activate` for the output.
- Dropout: `tk_dropout_forward(y, mask, n, rate, &rng)` /
  `tk_dropout_backward(dy, mask, n, rate)`.
- `tk_rng rng = tk_rng_seed(seed)` drives dropout masks and stochastic rounding.

Worked end-to-end trainer: [`examples/train_xor.c`](../examples/train_xor.c)
(`make train`). Gradient check: `make testbp`.

## Which functions to call

| You want to… | Call |
|---|---|
| Quantize a float weight into storage | `TK_FROM_FLOAT` / `tk_float_to_*` |
| Read a stored weight back as float | `TK_TO_FLOAT` / `tk_*_to_float` |
| One neuron / a dense layer forward | `tk_linear_forward` |
| Many inputs through one layer (GEMM) | `tk_linear_forward_batch` |
| Just a dot product | `tk_dot` |
| Apply an activation over a vector | `tk_activate` |
| Loss + seed gradient | `tk_loss` |
| Dense layer backward | `tk_linear_backward` |
| Update weights (SGD, L1/L2, SR) | `tk_sgd_step` / `tk_optim_default` |
| One float32 training step (FFI-friendly) | `tk_train_step_f32` |
| A whole epoch in one call (amortizes FFI) | `tk_train_epoch_f32` |
| Dropout forward / backward | `tk_dropout_forward` / `tk_dropout_backward` |
| Know the active dtype from Python | `Mantissa().dtype` |
