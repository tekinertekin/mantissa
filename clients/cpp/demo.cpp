// Use the downloaded mantissa shared library from C++ via runtime loading
// (dlopen / LoadLibrary) — no headers needed, just the library file.
//
//   forward pass (OR-gate perceptron) + a simple back-prop training loop.
//
// Build & run (macOS/Linux):
//   c++ -std=c++17 demo.cpp -o demo            # add -ldl on Linux
//   ./demo ../../dist/libmantissa.dylib        # or your downloaded file
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
  #include <windows.h>
  static void *load(const char *p){ return (void*)LoadLibraryA(p); }
  static void *sym(void *h, const char *n){ return (void*)GetProcAddress((HMODULE)h, n); }
  #define DEFAULT_LIB "libmantissa-windows-x86_64.dll"
#else
  #include <dlfcn.h>
  static void *load(const char *p){ return dlopen(p, RTLD_NOW); }
  static void *sym(void *h, const char *n){ return dlsym(h, n); }
  #if defined(__APPLE__)
    #define DEFAULT_LIB "libmantissa-macos-arm64.dylib"
  #else
    #define DEFAULT_LIB "libmantissa-linux-x86_64.so"
  #endif
#endif

// Activation ids (see activations.h): IDENTITY=0, STEP=1.
using dtype_name_fn = const char *(*)();
using forward_fn    = void (*)(const float*, const float*, const float*, float*, int, int, int);
using train_fn      = float (*)(float*, float*, const float*, const float*, int, int, int, float);

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1]
                     : (getenv("MANTISSA_LIB") ? getenv("MANTISSA_LIB") : DEFAULT_LIB);
    void *h = load(path);
    if (!h) { fprintf(stderr, "cannot load %s\n", path); return 1; }

    auto dtype_name = (dtype_name_fn)sym(h, "tk_dtype_name");
    auto forward    = (forward_fn)   sym(h, "tk_linear_forward_f32");
    auto train      = (train_fn)     sym(h, "tk_train_step_f32");
    if (!dtype_name || !forward || !train) {
        fprintf(stderr, "%s does not export the mantissa API\n", path);
        return 1;
    }

    printf("backend dtype: %s\n", dtype_name());

    // Forward: 2-input OR gate, y = step(w.x + b)
    float W[2] = {1, 1}, bias[1] = {-0.5f};
    float in[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    printf("OR perceptron:\n");
    for (int i = 0; i < 4; i++) {
        float y;
        forward(W, in[i], bias, &y, 1, 2, /*STEP*/1);
        printf("  (%g,%g) -> %g\n", in[i][0], in[i][1], y);
    }

    // Back-prop: learn a linear neuron mapping [1,2,3] -> 14
    float Wt[3] = {0,0,0}, bt[1] = {0};
    float x[3] = {1,2,3}, target[1] = {14};
    printf("training a linear neuron:\n");
    for (int s = 0; s <= 200; s++) {
        float loss = train(Wt, bt, x, target, 1, 3, /*IDENTITY*/0, 0.01f);
        if (s % 50 == 0) printf("  step %3d  loss %.6f\n", s, loss);
    }
    float y; forward(Wt, x, bt, &y, 1, 3, 0);
    printf("final prediction %.3f (target 14)\n", y);
    return 0;
}
