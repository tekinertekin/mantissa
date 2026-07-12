#include "backprop.h"
#include "config.h"
#include "pool.h"
#include <math.h>
#include <stdlib.h>

/* Backward for output rows [o0, o1): writes each row's dW/db (disjoint, so
 * thread-safe) and, if dxp != NULL, accumulates the input-gradient contribution
 * into dxp. dW row and dx share one pass over the row (wr/dWr sequential). */
static void bwd_range(const tk_scalar_t *restrict W, const tk_scalar_t *restrict x,
                      const float *restrict z, const float *restrict dy,
                      float *restrict dW, float *restrict db, float *restrict dxp,
                      int o0, int o1, int in_dim, tk_activation_t act) {
    for (int o = o0; o < o1; o++) {
        const float dz = dy[o] * tk_act_grad_scalar(z[o], act);
        if (db) db[o] = dz;
        const tk_scalar_t *restrict wr  = W  + (size_t)o * in_dim;
        float             *restrict dWr = dW + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++) {
            dWr[i] = dz * TK_TO_FLOAT(x[i]);
            if (dxp) dxp[i] += TK_TO_FLOAT(wr[i]) * dz;
        }
    }
}

typedef struct {
    const tk_scalar_t *W, *x;
    const float *z, *dy;
    float *dW, *db, *dx_partials;   /* per-thread dx accumulators (T x in_dim), or NULL */
    int in_dim;
    tk_activation_t act;
} tk_bwd_ctx;

static void bwd_worker(void *p, int o0, int o1, int worker) {
    const tk_bwd_ctx *c = (const tk_bwd_ctx *)p;
    float *dxp = c->dx_partials ? c->dx_partials + (size_t)worker * c->in_dim : NULL;
    bwd_range(c->W, c->x, c->z, c->dy, c->dW, c->db, dxp, o0, o1, c->in_dim, c->act);
}

void tk_linear_backward(const tk_scalar_t *restrict W,
                        const tk_scalar_t *restrict x,
                        const float *restrict z,
                        const float *restrict dy,
                        float *restrict dW,
                        float *restrict db,
                        float *restrict dx,
                        int out_dim, int in_dim,
                        tk_activation_t act) {
    const int T = ((size_t)out_dim * in_dim >= TK_MT_MIN_WORK) ? tk_num_threads() : 1;

    if (T > 1 && !dx) {                       /* no input gradient: pure row-parallel */
        tk_bwd_ctx c = { W, x, z, dy, dW, db, NULL, in_dim, act };
        tk_parallel_for(out_dim, bwd_worker, &c);
        return;
    }
    if (T > 1) {                              /* dx is a reduction over rows */
        float *partials = calloc((size_t)T * in_dim, sizeof(float));
        if (partials) {                       /* each worker accumulates privately... */
            tk_bwd_ctx c = { W, x, z, dy, dW, db, partials, in_dim, act };
            tk_parallel_for(out_dim, bwd_worker, &c);
            for (int i = 0; i < in_dim; i++) {  /* ...then sum the partials into dx */
                float s = 0.0f;
                for (int t = 0; t < T; t++) s += partials[(size_t)t * in_dim + i];
                dx[i] = s;
            }
            free(partials);
            return;
        }                                     /* allocation failed -> serial */
    }

    if (dx) for (int i = 0; i < in_dim; i++) dx[i] = 0.0f;
    bwd_range(W, x, z, dy, dW, db, dx, 0, out_dim, in_dim, act);
}

tk_optim tk_optim_default(float lr) {
    tk_optim o;
    o.lr         = lr;
    o.l1         = TK_USE_L1 ? (float)(TK_L1_LAMBDA) : 0.0f;
    o.l2         = TK_USE_L2 ? (float)(TK_L2_LAMBDA) : 0.0f;
    o.stochastic = TK_USE_STOCHASTIC_ROUNDING;
    return o;
}

/* Stochastic rounding to the active storage grid. The ULP at v is built by bit
 * arithmetic on the exponent (biased_exp - TK_MANT_BITS) instead of
 * frexpf/ldexpf, dropping two libm calls per weight; rounding onto the grid
 * keeps the final TK_FROM_FLOAT exact. */
static inline tk_scalar_t tk_sr_from_float(float v, tk_rng *rng) {
    const uint32_t bits = tk__f2u(v);
    const uint32_t bexp = (bits >> 23) & 0xFFu;
    if (v == 0.0f || bexp == 0xFFu || bexp <= (uint32_t)TK_MANT_BITS)
        return TK_FROM_FLOAT(v);                        /* 0 / inf / nan / tiny */
    const float ulp    = tk__u2f((bexp - (uint32_t)TK_MANT_BITS) << 23);
    const float scaled = v / ulp;
    const float fl     = floorf(scaled);
    const float chosen = (tk_rng_f01(rng) < (scaled - fl)) ? fl + 1.0f : fl;
    return TK_FROM_FLOAT(chosen * ulp);
}

void tk_sgd_step(tk_scalar_t *restrict W, const float *restrict dW,
                 int n, const tk_optim *opt, tk_rng *rng) {
    const float lr = opt->lr, l1 = opt->l1, l2 = opt->l2;
    const int sr = opt->stochastic && rng;
    tk_rng r = sr ? *rng : (tk_rng){0};                 /* keep RNG state in a register */
    for (int i = 0; i < n; i++) {
        float w = TK_TO_FLOAT(W[i]);
        float g = dW[i];
        if (l2 != 0.0f) g += l2 * w;
        if (l1 != 0.0f) g += l1 * (w > 0.0f ? 1.0f : (w < 0.0f ? -1.0f : 0.0f));
        w -= lr * g;
        W[i] = sr ? tk_sr_from_float(w, &r) : TK_FROM_FLOAT(w);
    }
    if (sr) *rng = r;
}

void tk_dropout_forward(float *restrict y, uint8_t *restrict mask,
                        int n, float rate, tk_rng *rng) {
    const float scale = 1.0f / (1.0f - rate);
    tk_rng r = *rng;                                    /* hoist state out of the loop */
    for (int i = 0; i < n; i++) {
        uint8_t keep = tk_rng_f01(&r) >= rate;
        mask[i] = keep;
        y[i]    = keep ? y[i] * scale : 0.0f;
    }
    *rng = r;
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
    if (out_dim <= 0) return 0.0f;                      /* empty layer: 0, not NaN */
    const float inv = 1.0f / (float)out_dim;
    float loss = 0.0f;
    for (int o = 0; o < out_dim; o++) {
        float *restrict wr = W + (size_t)o * in_dim;
        /* Four independent accumulators break the loop-carried FP-add dependency
         * so the forward dot pipelines/vectorizes (same recipe as tk_dot). */
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= in_dim; i += 4) {
            s0 += wr[i + 0] * x[i + 0];
            s1 += wr[i + 1] * x[i + 1];
            s2 += wr[i + 2] * x[i + 2];
            s3 += wr[i + 3] * x[i + 3];
        }
        float z = (bias ? bias[o] : 0.0f) + ((s0 + s1) + (s2 + s3));
        for (; i < in_dim; i++) z += wr[i] * x[i];   /* forward tail */
        float y = tk_act_scalar(z, act);
        float d = y - target[o];
        loss += d * d;
        float dz = 2.0f * d * inv * tk_act_grad_scalar(z, act); /* dL/dz */
        if (dz != 0.0f) {                       /* dead row (e.g. relu z<=0): the
                                                 * update is a no-op; skip the
                                                 * whole read-modify-write pass */
            for (int i = 0; i < in_dim; i++) wr[i] -= lr * dz * x[i];
            if (bias) bias[o] -= lr * dz;
        }
    }
    return loss * inv;
}

float tk_train_epoch_f32(float *W, float *bias,
                         const float *X, const float *targets,
                         int n_samples, int out_dim, int in_dim,
                         tk_activation_t act, float lr) {
    if (n_samples <= 0) return 0.0f;
    double sum = 0.0;                           /* mean over n: f64 keeps it exact */
    for (int s = 0; s < n_samples; s++)
        sum += tk_train_step_f32(W, bias,
                                 X + (size_t)s * in_dim,
                                 targets + (size_t)s * out_dim,
                                 out_dim, in_dim, act, lr);
    return (float)(sum / n_samples);
}
