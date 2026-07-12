#ifndef MANTISSA_BACKPROP_H
#define MANTISSA_BACKPROP_H

#include <stdint.h>
#include "dtypes.h"
#include "activations.h"
#include "tk_export.h"

/* Reverse pass of the dense layer, plus the pieces training needs: an
 * optimizer step (with optional L1/L2 and stochastic rounding) and dropout.
 * No autograd graph -- the caller drives the layers, matching the explicit,
 * inspectable style of the forward core. */

/* --- fast PRNG (xorshift64) for dropout masks and stochastic rounding ------ */
typedef struct { uint64_t s; } tk_rng;

static inline tk_rng tk_rng_seed(uint64_t seed) {
    tk_rng r; r.s = seed ? seed : 0x9E3779B97F4A7C15ull;   /* never zero */
    return r;
}
static inline uint64_t tk_rng_u64(tk_rng *r) {
    uint64_t x = r->s; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return r->s = x;
}
static inline float tk_rng_f01(tk_rng *r) {                /* [0,1) */
    return (float)(tk_rng_u64(r) >> 40) * (1.0f / 16777216.0f);
}

/* --- dense layer backward --------------------------------------------------
 * Given the cached forward input x and pre-activation z, and the incoming
 * output gradient dy, compute:
 *   dW[o*in+i] = dz[o] * x[i]         (weight gradient)
 *   db[o]      = dz[o]                (bias gradient; skip if db == NULL)
 *   dx[i]      = sum_o W[o*in+i]*dz[o] (input gradient; skip if dx == NULL)
 * where dz[o] = dy[o] * act'(z[o]). */
TK_API void tk_linear_backward(const tk_scalar_t *restrict W,
                               const tk_scalar_t *restrict x,
                               const float *restrict z,
                               const float *restrict dy,
                               float *restrict dW,
                               float *restrict db,
                               float *restrict dx,
                               int out_dim, int in_dim,
                               tk_activation_t act);

/* --- optimizer ------------------------------------------------------------- */
typedef struct {
    float lr;          /* learning rate */
    float l1;          /* L1 strength (0 = off) */
    float l2;          /* L2 strength (0 = off) */
    int   stochastic;  /* 1 = stochastic-round the weight write-back */
} tk_optim;

/* Optimizer prefilled from config.h (l1/l2/stochastic default to off). */
TK_API tk_optim tk_optim_default(float lr);

/* In-place SGD update of narrow-stored weights: W -= lr*(dW + reg). The updated
 * float is written back through the storage type, optionally with stochastic
 * rounding (rng may be NULL when stochastic == 0). */
TK_API void tk_sgd_step(tk_scalar_t *restrict W, const float *restrict dW,
                        int n, const tk_optim *opt, tk_rng *rng);

/* --- dropout (inverted; scales survivors by 1/(1-rate)) -------------------- */
TK_API void tk_dropout_forward(float *restrict y, uint8_t *restrict mask,
                               int n, float rate, tk_rng *rng);
TK_API void tk_dropout_backward(float *restrict dy, const uint8_t *restrict mask,
                                int n, float rate);

/* --- convenience: one float32 training step for a single dense layer -------
 * Forward + MSE loss + backward + in-place SGD update, all in plain float32.
 * A single call marshals only float arrays and scalars, so any language with a
 * C FFI (Python, C#, Java, Node, Rust, ...) can train with it directly. Returns
 * the pre-update MSE loss; `bias` may be NULL. For the full narrow-storage /
 * stochastic-rounding path use tk_linear_backward + tk_sgd_step. */
TK_API float tk_train_step_f32(float *W, float *bias,
                               const float *x, const float *target,
                               int out_dim, int in_dim,
                               tk_activation_t act, float lr);

/* One full epoch of sequential SGD, driven inside the library: X is n_samples
 * rows of in_dim, targets n_samples rows of out_dim. Weight updates are
 * numerically identical to calling tk_train_step_f32 once per sample — the
 * point is one FFI crossing per epoch instead of one per sample (an
 * interpreted caller pays microseconds per crossing, which dominates small
 * layers). Returns the mean pre-update loss over the epoch. */
TK_API float tk_train_epoch_f32(float *W, float *bias,
                                const float *X, const float *targets,
                                int n_samples, int out_dim, int in_dim,
                                tk_activation_t act, float lr);

#endif /* MANTISSA_BACKPROP_H */
