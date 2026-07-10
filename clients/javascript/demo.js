// Use the downloaded mantissa shared library from Node.js via koffi (FFI).
//
//   npm install koffi
//   node demo.js ../../dist/libmantissa.dylib      # or your downloaded file
//
// Shows a forward pass (OR-gate perceptron) and a simple back-prop training loop.
const koffi = require("koffi");

function defaultLib() {
  switch (process.platform) {
    case "darwin": return "libmantissa-macos-arm64.dylib";
    case "win32":  return "libmantissa-windows-x86_64.dll";
    default:       return "libmantissa-linux-x86_64.so";
  }
}

const path = process.argv[2] || process.env.MANTISSA_LIB || defaultLib();
const lib = koffi.load(path);

// Activation ids (activations.h): IDENTITY=0, STEP=1.
const dtypeName = lib.func("const char* tk_dtype_name()");
const forward = lib.func(
  "void tk_linear_forward_f32(float *W, float *x, float *bias, _Out_ float *y, int outDim, int inDim, int act)");
const trainStep = lib.func(
  "float tk_train_step_f32(_Inout_ float *W, _Inout_ float *bias, float *x, float *target, int outDim, int inDim, int act, float lr)");

console.log("backend dtype:", dtypeName());

// Forward: 2-input OR gate
const W = [1, 1], bias = [-0.5];
console.log("OR perceptron:");
for (const xin of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  const y = [0];
  forward(W, xin, bias, y, 1, 2, /*STEP*/ 1);
  console.log(`  (${xin}) -> ${y[0]}`);
}

// Back-prop: learn a linear neuron mapping [1,2,3] -> 14
const Wt = [0, 0, 0], bt = [0];
const x = [1, 2, 3], target = [14];
console.log("training a linear neuron:");
for (let s = 0; s <= 200; s++) {
  const loss = trainStep(Wt, bt, x, target, 1, 3, /*IDENTITY*/ 0, 0.01);
  if (s % 50 === 0) console.log(`  step ${s}  loss ${loss.toFixed(6)}`);
}
const y = [0];
forward(Wt, x, bt, y, 1, 3, 0);
console.log(`final prediction ${y[0].toFixed(3)} (target 14)`);
