#ifndef MANTISSA_ACTIVATIONS_H
#define MANTISSA_ACTIVATIONS_H

#include <stdint.h>
#include <string.h>
#include "tk_export.h"

/* Reinterpret a float's bits (no aliasing violation, no-op at -O2). */
static inline uint32_t tk_act__bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

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
        /* Read the sign bit directly instead of comparing: measured ~40% faster
         * than `z >= 0 ? 1 : 0` in a vectorized loop (see bench). */
        case TK_ACT_STEP:    return 1.0f - (float)(tk_act__bits(z) >> 31);
        /* copysign / fmax lower to a single bitwise/max instruction and
         * vectorize; branchless. */
        case TK_ACT_SIGN:    return __builtin_copysignf(1.0f, z);
        case TK_ACT_RELU:    return __builtin_fmaxf(z, 0.0f);
        case TK_ACT_SIGMOID: return 1.0f / (1.0f + __builtin_expf(-z));
        case TK_ACT_TANH:    return __builtin_tanhf(z);
        case TK_ACT_GELU:    return 0.5f * z * (1.0f + __builtin_tanhf(
                                    0.7978845608028654f * (z + 0.044715f * z * z * z)));
        default:             return z;
    }
}

/* Apply an activation over a contiguous vector in place (resolves once). */
TK_API void tk_activate(float *y, int n, tk_activation_t a);

/* Derivative d(act)/dz at pre-activation z, for back-propagation.
 * step/sign are non-differentiable (gradient 0) and only make sense in a
 * forward-only perceptron. */
static inline float tk_act_grad_scalar(float z, tk_activation_t a) {
    switch (a) {
        case TK_ACT_RELU:    return z > 0.0f ? 1.0f : 0.0f;
        case TK_ACT_SIGMOID: { float s = 1.0f / (1.0f + __builtin_expf(-z));
                               return s * (1.0f - s); }
        case TK_ACT_TANH:    { float t = __builtin_tanhf(z); return 1.0f - t * t; }
        case TK_ACT_GELU: {
            const float c = 0.7978845608028654f, a3 = 0.044715f;
            float u  = c * (z + a3 * z * z * z);
            float th = __builtin_tanhf(u);
            float du = c * (1.0f + 3.0f * a3 * z * z);
            return 0.5f * (1.0f + th) + 0.5f * z * (1.0f - th * th) * du;
        }
        case TK_ACT_IDENTITY: return 1.0f;
        default:              return 0.0f;   /* step, sign */
    }
}

#endif /* MANTISSA_ACTIVATIONS_H */
