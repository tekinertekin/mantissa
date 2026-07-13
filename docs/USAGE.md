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

A shuffling caller passes its permutation as `order=` (int32, zero-copy)
instead of materializing row-permuted copies of `X`/`targets` every epoch —
same sequence, bit-identical weights. `mistakes=True` additionally returns the
in-epoch pre-update mistake count (rows where `target*z <= 0` as visited) as a
`(loss, mistakes)` tuple, saving a separate full forward pass per epoch — note
it is not the same number as a post-epoch pass with the final weights:

```python
order = rng.permutation(1000).astype(np.int32)
loss, mistakes = tk.train_epoch(W, X, targets, 1000, 1, 4, IDENTITY, 0.01,
                                bias=bias, order=order, mistakes=True)
```

The classic mistake-driven perceptron rule (Rosenblatt, 1958; targets ±1,
update only on `target*z <= 0`) is its own one-call epoch,
`tk.perceptron_epoch` (wrapping `tk_perceptron_epoch_f32`) — returns the
epoch's mistake count, 0 meaning the data was separated this pass (measured
~314x over the per-sample `linear_forward` + numpy-update loop at 1030x4x100
epochs):

```python
mistakes = tk.perceptron_epoch(W, X, targets, n_samples=1000, out_dim=1,
                               in_dim=4, lr=0.01, bias=bias, order=order)
```

When those epoch calls sit in a loop — the normal case — bind the session once
with `tk.trainer()`. Measured (M4, 1030×4): a `perceptron_epoch` wrapper call
costs ~9.8 µs of which only ~3 µs is the C epoch; the rest re-derives ctypes
pointers for the same five unchanged buffers every epoch. `Trainer` derives
them once (**~2.1x per epoch call**; ~1.4x on the heavier ordered SGD epoch),
same C entry points, bit-identical weight trajectories. Buffers must be
C-contiguous float32 and are mutated in place. `margins(out)` writes the
no-bias linear responses `w·x_s` for every sample in one row-parallel GEMV
(out_dim 1 only) — the post-epoch convergence count made cheap (~2x over
`linear_forward` with per-call binding):

```python
trainer = tk.trainer(W, X, targets, n_samples=1000, out_dim=1, in_dim=4,
                     bias=bias)                       # bind pointers once
z = np.empty(1000, dtype=np.float32)
for epoch in range(100):
    order = rng.permutation(1000).astype(np.int32)
    mistakes = trainer.perceptron_epoch(0.01, order=order)
    # or: loss = trainer.train_epoch(IDENTITY, 0.01, order=order)
    #     trainer.margins(z)   # z[s] = w . x_s  (add your bias scalar)
    if mistakes == 0:
        break
```

The CNN-primitive family has the same pattern one level up: `tk.session()`
returns a view with identical method signatures whose pointer conversions are
memoized by array identity — built for training loops where parameters and
scratch are allocated once and refilled in place (the mantissa-cnn shape).
Measured on a LeNet-5/MNIST fit (M4): pointer conversion was ~12% of wall
time; the session removes it (268 → 234 ms end to end). One session per
model — the memo pins its arrays alive.

For repeated *inference*, pass `out=` to `linear_forward` (a float32 numpy
array or `array('f')` of `out_dim`): the result is written straight into your
buffer — no per-call output allocation or boxing (~1.24x per call at 256x256).

For repeated inference on **fixed weights**, `tk.prepare()` pre-quantizes the
weights into the storage dtype once and holds them resident narrow, so each
`forward()` skips re-quantizing every weight and runs the narrow SIMD kernel
directly (`linear_forward` re-quantizes W on every call). Measured from Python
(bf16, serial): ~2.3x at 64x64, ~7x at 256x256, ~16x at 1024x1024, at half the
weight bytes. Results are bit-identical to `tk_linear_forward`.

```python
layer = tk.prepare(W, out_dim=256, in_dim=256, bias=bias)   # narrow once
for x in stream:
    y = layer.forward(x, act=RELU, out=out)                 # cheap per call
```

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

## Convolution (CNN primitives)

The conv family (`conv.h`, since v0.2.1) is pure **float32** end to end — no
storage-dtype quantization on the training path (narrow-storage conv is a
roadmap item). Layout is **NCHW**, row-major, batch outermost; every buffer is
caller-allocated. Convolution is im2col + a register-blocked GEMM (Chellapilla
et al., 2006), one sample's patch matrix at a time, threaded across the batch.
Max pooling uses floor semantics (no padding: `out = (in - pool)/stride + 1`,
ragged edges dropped) and hands the backward its argmax indices.

```c
#include "conv.h"

/* one conv layer + softmax head, forward and backward */
int oh = tk_conv2d_out_dim(28, 5, /*stride*/1, /*pad*/0);   /* 24 */
tk_conv2d_forward_f32(X, K, bias, Z, Y, n, 1, 28, 28,       /* X: n x 1x28x28 */
                      6, 5, 5, 1, 0, TK_ACT_RELU);          /* -> n x 6x24x24 */
tk_maxpool2d_f32(Y, P, argmax, n, 6, oh, oh, 2, 2);         /* -> n x 6x12x12 */
tk_linear_forward_batch_f32(W, Pflat, wb, Zl, logits, n, 10, 6*12*12,
                            TK_ACT_IDENTITY);
float loss = tk_softmax_xent_f32(logits, labels, dlogits, n, 10);

tk_linear_backward_batch_f32(W, Pflat, Zl, dlogits, dW, dwb, dP,
                             n, 10, 6*12*12, TK_ACT_IDENTITY);
tk_maxpool2d_backward_f32(dP, argmax, dY, n, 6, oh, oh, 12, 12);
tk_conv2d_backward_f32(X, K, Z, dY, dK, db, NULL /*first layer*/,
                       n, 1, 28, 28, 6, 5, 5, 1, 0, TK_ACT_RELU);
tk_sgd_update_f32(K, dK, 6*1*5*5, 0.05f);                   /* plain f32 SGD */
```

`Z` is the saved pre-activation the backward needs (pass `NULL` with
`TK_ACT_IDENTITY` if you only want `Y`); `dK`/`db` come back summed over the
batch. Gradient checks: `make testconv`. Shapes bench: `make benchconv` —
measured (M4, float32): VGG-block 64×32×32 → 64@3×3 forward 24.4 ms serial /
6.1 ms threaded (199 GFLOP/s), backward 52.6 / 14.8 ms.

The same family from Python (zero-copy on C-contiguous float32 / int32 numpy
arrays; feature-detect with `hasattr(tk, "conv2d_forward")`):

```python
import numpy as np
from mantissa import Mantissa, RELU, IDENTITY

tk = Mantissa()
n, oh = 32, tk.conv2d_out_dim(28, 5, 1, 0)          # 24
X  = np.random.rand(n * 1 * 28 * 28).astype(np.float32)
K  = (np.random.rand(6 * 1 * 5 * 5).astype(np.float32) - 0.5) * 0.2
kb = np.zeros(6, np.float32)
Z  = np.empty(n * 6 * oh * oh, np.float32); Y = np.empty_like(Z)

tk.conv2d_forward(X, K, kb, Z, Y, n, 1, 28, 28, 6, 5, 5, 1, 0, RELU)
# ... maxpool2d / linear_forward_batch / softmax_xent mirror the C calls ...
dY = np.empty_like(Y); dK = np.empty_like(K); db = np.empty_like(kb)
tk.conv2d_backward(X, K, Z, dY, dK, db, None, n, 1, 28, 28, 6, 5, 5, 1, 0, RELU)
tk.sgd_update(K, dK, K.size, 0.05)                  # in place
```

Runnable cross-check: [`python/test_conv_binding.py`](../python/test_conv_binding.py)
(conv vs a numpy im2col reference, plus the whole family on random data).

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
| A shuffled epoch without copying rows, in-epoch mistake count | `tk_train_epoch_order_f32` |
| An epoch of the Rosenblatt perceptron rule (returns mistakes) | `tk_perceptron_epoch_f32` |
| Epoch calls in a loop without per-call pointer rebinding (Python) | `Mantissa().trainer(...)` |
| Repeated inference on fixed weights (Python) | `Mantissa().prepare(...).forward(...)` |
| Dropout forward / backward | `tk_dropout_forward` / `tk_dropout_backward` |
| Conv layer forward, batched float32 (NCHW) | `tk_conv2d_forward_f32` |
| Conv layer backward (dK/db/dX, batch-summed) | `tk_conv2d_backward_f32` |
| Max pooling forward / backward (argmax scatter) | `tk_maxpool2d_f32` / `tk_maxpool2d_backward_f32` |
| A whole batch through a float32 dense layer (CNN head) | `tk_linear_forward_batch_f32` / `tk_linear_backward_batch_f32` |
| Softmax + cross-entropy, fused (loss + dlogits) | `tk_softmax_xent_f32` |
| Plain float32 SGD update | `tk_sgd_update_f32` |
| Conv/pool output spatial size | `tk_conv2d_out_dim` |
| Know the active dtype from Python | `Mantissa().dtype` |
