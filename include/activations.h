#ifndef MANTISSA_ACTIVATIONS_H
#define MANTISSA_ACTIVATIONS_H

#include "tk_export.h"

/* Shared by every layer type this library will feed: perceptron (step/sign),
 * MLP/RNN (relu/tanh/sigmoid), LSTM/GRU gates (sigmoid/tanh), Transformer (gelu). */
typedef enum {
    TK_ACT_IDENTITY = 0,
    TK_ACT_STEP     = 1,   /* 1 if z >= 0 else 0        (Rosenblatt perceptron) */
    TK_ACT_SIGN     = 2,   /* +1 if z >= 0 else -1 */
    TK_ACT_RELU     = 3,   /* max(0, z) */
    TK_ACT_SIGMOID  = 4,   /* 1 / (1 + e^-z) */
    TK_ACT_TANH     = 5,
    TK_ACT_GELU     = 6,   /* tanh approximation (Hendrycks & Gimpel, 2016, arXiv:1606.08415) */
    TK_ACT_COUNT    = 7
} tk_activation_t;

/* Individual functions, so a caller can bind one directly and skip dispatch. */
TK_API float tk_act_identity(float z);
TK_API float tk_act_step(float z);
TK_API float tk_act_sign(float z);
TK_API float tk_act_relu(float z);
TK_API float tk_act_sigmoid(float z);
TK_API float tk_act_tanh(float z);
TK_API float tk_act_gelu(float z);

/* Function-pointer dispatch. Resolve ONCE per layer/vector, then call the
 * pointer in the loop -- this replaces a per-element `switch` (whose jump
 * table is hit on every element) with a single lookup. See docs/DESIGN.md and
 * `make bench` for the measured difference. */
typedef float (*tk_act_fn)(float);
TK_API tk_act_fn tk_act_resolve(tk_activation_t a);

/* Inline switch kept for the per-neuron path (a layer applies the activation
 * out_dim times, not per multiply-add) and as the baseline in the benchmark. */
static inline float tk_act_scalar(float z, tk_activation_t a) {
    switch (a) {
        case TK_ACT_STEP:    return z >= 0.0f ? 1.0f : 0.0f;
        case TK_ACT_SIGN:    return z >= 0.0f ? 1.0f : -1.0f;
        case TK_ACT_RELU:    return z > 0.0f ? z : 0.0f;
        case TK_ACT_SIGMOID: return 1.0f / (1.0f + __builtin_expf(-z));
        case TK_ACT_TANH:    return __builtin_tanhf(z);
        case TK_ACT_GELU:    return 0.5f * z * (1.0f + __builtin_tanhf(
                                    0.7978845608028654f * (z + 0.044715f * z * z * z)));
        default:             return z;
    }
}

/* Apply an activation over a contiguous vector in place (resolves once). */
TK_API void tk_activate(float *y, int n, tk_activation_t a);

#endif /* MANTISSA_ACTIVATIONS_H */
