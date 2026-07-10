#include "backprop.h"
#include "config.h"
#include <math.h>

void tk_linear_backward(const tk_scalar_t *restrict W,
                        const tk_scalar_t *restrict x,
                        const float *restrict z,
                        const float *restrict dy,
                        float *restrict dW,
                        float *restrict db,
                        float *restrict dx,
                        int out_dim, int in_dim,
                        tk_activation_t act) {
    if (dx) for (int i = 0; i < in_dim; i++) dx[i] = 0.0f;

    for (int o = 0; o < out_dim; o++) {
        /* Collapse the activation derivative into the output gradient once. */
        const float dz = dy[o] * tk_act_grad_scalar(z[o], act);
        if (db) db[o] = dz;

        const tk_scalar_t *restrict wr  = W  + (size_t)o * in_dim;
        float             *restrict dWr = dW + (size_t)o * in_dim;
        /* dW row and dx accumulation share one pass over the row: wr/dWr are
         * sequential (cache-friendly); dx stays hot across the o loop. */
        for (int i = 0; i < in_dim; i++) {
            dWr[i] = dz * TK_TO_FLOAT(x[i]);
            if (dx) dx[i] += TK_TO_FLOAT(wr[i]) * dz;
        }
    }
}

tk_optim tk_optim_default(float lr) {
    tk_optim o;
    o.lr         = lr;
    o.l1         = TK_USE_L1 ? (float)(TK_L1_LAMBDA) : 0.0f;
    o.l2         = TK_USE_L2 ? (float)(TK_L2_LAMBDA) : 0.0f;
    o.stochastic = TK_USE_STOCHASTIC_ROUNDING;
    return o;
}

/* Stochastic rounding to the active storage grid. ulp is derived from the
 * value's binade and the type's mantissa width (TK_MANT_BITS); rounding onto
 * the grid makes the final TK_FROM_FLOAT exact. */
static inline tk_scalar_t tk_sr_from_float(float v, tk_rng *rng) {
    if (v == 0.0f || !isfinite(v)) return TK_FROM_FLOAT(v);
    int e; frexpf(fabsf(v), &e);                       /* |v| in [2^(e-1), 2^e) */
    float ulp    = ldexpf(1.0f, (e - 1) - TK_MANT_BITS);
    float scaled = v / ulp;
    float fl     = floorf(scaled);
    float chosen = (tk_rng_f01(rng) < (scaled - fl)) ? fl + 1.0f : fl;
    return TK_FROM_FLOAT(chosen * ulp);
}

void tk_sgd_step(tk_scalar_t *restrict W, const float *restrict dW,
                 int n, const tk_optim *opt, tk_rng *rng) {
    const float lr = opt->lr, l1 = opt->l1, l2 = opt->l2;
    for (int i = 0; i < n; i++) {
        float w = TK_TO_FLOAT(W[i]);
        float g = dW[i];
        if (l2 != 0.0f) g += l2 * w;
        if (l1 != 0.0f) g += l1 * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
        w -= lr * g;
        W[i] = (opt->stochastic && rng) ? tk_sr_from_float(w, rng) : TK_FROM_FLOAT(w);
    }
}

void tk_dropout_forward(float *restrict y, uint8_t *restrict mask,
                        int n, float rate, tk_rng *rng) {
    const float scale = 1.0f / (1.0f - rate);
    for (int i = 0; i < n; i++) {
        uint8_t keep = tk_rng_f01(rng) >= rate;
        mask[i] = keep;
        y[i]    = keep ? y[i] * scale : 0.0f;
    }
}

void tk_dropout_backward(float *restrict dy, const uint8_t *restrict mask,
                         int n, float rate) {
    const float scale = 1.0f / (1.0f - rate);
    for (int i = 0; i < n; i++) dy[i] = mask[i] ? dy[i] * scale : 0.0f;
}

float tk_train_step_f32(float *W, float *bias,
                        const float *x, const float *target,
                        int out_dim, int in_dim,
                        tk_activation_t act, float lr) {
    const float inv = 1.0f / (float)out_dim;
    float loss = 0.0f;
    for (int o = 0; o < out_dim; o++) {
        float *restrict wr = W + (size_t)o * in_dim;
        float z = bias ? bias[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) z += wr[i] * x[i];   /* forward */
        float y = tk_act_scalar(z, act);
        float d = y - target[o];
        loss += d * d;
        float dz = 2.0f * d * inv * tk_act_grad_scalar(z, act); /* dL/dz */
        for (int i = 0; i < in_dim; i++) wr[i] -= lr * dz * x[i]; /* SGD update */
        if (bias) bias[o] -= lr * dz;
    }
    return loss * inv;
}
