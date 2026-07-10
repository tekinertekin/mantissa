# Language clients

Using the prebuilt **mantissa** shared library from other languages — download
the library for your OS from the
[releases](https://github.com/tekinertekin/mantissa/releases) (or build it with
`make dist`), then run one of these. No headers or `make` needed by the caller.

Every example does the same three things, so you can compare the FFI styles:

1. print the backend dtype — `tk_dtype_name()`
2. a **forward pass** — a 2-input OR-gate perceptron via `tk_linear_forward_f32`
3. a **back-prop** training loop — learn a linear neuron (`[1,2,3] → 14`) with
   `tk_train_step_f32` (one call = forward + gradient + SGD update), loss → 0

| Language | FFI mechanism | Folder |
|----------|---------------|--------|
| Python     | `ctypes` (stdlib)        | [`../python`](../python) |
| C++        | `dlopen` / `LoadLibrary` | [`cpp`](cpp) |
| C#         | P/Invoke (`DllImport`)   | [`csharp`](csharp) |
| Java       | JNA                      | [`java`](java) |
| JavaScript | Node + `koffi`           | [`javascript`](javascript) |
| Rust       | `libloading`             | [`rust`](rust) |

All three functions take plain `float32` arrays, so marshalling stays trivial in
every language. Activation ids: `IDENTITY=0, STEP=1, SIGN=2, RELU=3, SIGMOID=4,
TANH=5, GELU=6`.

The C ABI is language-neutral, so others work the same way: **R** via `Rcpp`,
**Julia** via `ccall`, **Go** via `cgo` — declare the three prototypes and call.
