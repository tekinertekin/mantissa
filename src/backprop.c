#include "backprop.h"
#include "config.h"
#include "pool.h"
#include <stdlib.h>
#if defined(__aarch64__) && TK_DTYPE == TK_DTYPE_BFLOAT16
#include <arm_neon.h>   /* plain-SGD bf16 fast path in tk_sgd_step */
#endif

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

/* Out-of-line landing pad for SR's 0/tiny/inf/nan guard. Deliberately noinline:
 * if the guard path ends in the same TK_FROM_FLOAT call as the hot path, clang
 * tail-merges the two and if-converts the guard, parking a csel (the RNG-state
 * rollback) on the xorshift loop-carried chain -- measured -12% on bf16, whose
 * SR loop is bound by exactly that chain. A distinct call target forces a real,
 * predictable branch and keeps the chain at its 3-eor minimum. */
static tk_scalar_t __attribute__((noinline, cold)) tk_sr_guarded(float v) {
    return TK_FROM_FLOAT(v);
}

/* Stochastic rounding to the active storage grid, in pure bit arithmetic: add
 * a random k-bit tail below the target's mantissa width to the f32 pattern,
 * then truncate onto the grid (Gupta et al., 2015, arXiv:1502.02551). No
 * fdiv/floorf/int->float convert per weight.
 *
 * Unbiasedness. Let k = 23 - TK_MANT_BITS and f = bits mod 2^k. Grid points are
 * the f32 patterns with k low zeros; the two neighbours of v are down =
 * bits - f and up = down + 2^k. Since the f32 value is affine in the bit
 * integer within a binade with slope ulp32 = 2^(bexp-150), and a mantissa
 * carry into the exponent lands exactly on the next binade's first grid point
 * (still 2^k * ulp32 above `down` in value), val(up) - val(down) =
 * 2^k * ulp32 throughout. With r uniform on [0, 2^k), (bits + r) & ~(2^k - 1)
 * yields `up` iff f + r >= 2^k, i.e. with probability f/2^k, so
 *   E[result] = val(down) + (f/2^k)(val(up) - val(down)) = v   exactly.
 * (Negative values: the pattern is sign-magnitude, so the identical algebra
 * runs on |v|; the sign-conditional tail below only aligns the decision with
 * the previous float implementation, it does not affect the probability.)
 * Grid points carry <= TK_MANT_BITS+1 significand bits, so the final
 * TK_FROM_FLOAT is exact for target-normal results; in the target-subnormal
 * band it re-rounds RNE onto the coarser grid, as before. Draws exactly one
 * u64 per non-guarded weight and makes the same up/down decision as the old
 * fdiv/floorf code for every (value, RNG state), so seeded weight
 * trajectories are bit-identical across the change. */
static inline tk_scalar_t tk_sr_from_float(float v, tk_rng *rng) {
    const uint32_t bits = tk__f2u(v);
    const uint32_t bexp = (bits >> 23) & 0xFFu;
    /* 0 / tiny / inf / nan in one unsigned range test; bexp==0 covers v==0.
     * No RNG draw here, matching the pre-integer-SR stream. */
    if (bexp - ((uint32_t)TK_MANT_BITS + 1u) >= 0xFEu - (uint32_t)TK_MANT_BITS)
        return tk_sr_guarded(v);
#if TK_MANT_BITS >= 23
    (void)tk_rng_u64(rng);       /* grid is exact; draw keeps the stream aligned */
    return TK_FROM_FLOAT(v);
#else
    const uint32_t k    = 23u - (uint32_t)TK_MANT_BITS;
    const uint32_t mask = (1u << k) - 1u;
    const uint32_t t    = (uint32_t)(tk_rng_u64(rng) >> (64u - k));
    const uint32_t sgn  = (uint32_t)((int32_t)bits >> 31);   /* all-ones if v<0 */
    const uint32_t r    = (t ^ ~sgn) & mask;   /* mask-t / t: matches the old
                                                * floor-based decision either sign */
    return TK_FROM_FLOAT(tk__u2f((bits + r) & ~mask));
#endif
}

#if defined(__aarch64__) && TK_DTYPE == TK_DTYPE_BFLOAT16
/* float32x4 -> bf16x4, bit-identical to tk_float_to_bf16 (RNE via +0x7FFF plus
 * the odd bit; NaN quieted by setting bit 22, i.e. bit 6 of the bf16). Verified
 * against the scalar converter over all 2^32 input patterns. */
static inline uint16x4_t tk_bf16_rne_narrow(float32x4_t w) {
    const uint32x4_t x    = vreinterpretq_u32_f32(w);
    const uint32x4_t absx = vandq_u32(x, vdupq_n_u32(0x7FFFFFFFu));
    const uint32x4_t nan  = vcgtq_u32(absx, vdupq_n_u32(0x7F800000u));
    const uint32x4_t odd  = vandq_u32(vshrq_n_u32(x, 16), vdupq_n_u32(1u));
    const uint32x4_t rne  = vaddq_u32(vaddq_u32(x, vdupq_n_u32(0x7FFFu)), odd);
    const uint32x4_t nanv = vorrq_u32(x, vdupq_n_u32(0x00400000u));
    return vshrn_n_u32(vbslq_u32(nan, nanv, rne), 16);
}
#endif

void tk_sgd_step(tk_scalar_t *restrict W, const float *restrict dW,
                 int n, const tk_optim *opt, tk_rng *rng) {
    const float lr = opt->lr, l1 = opt->l1, l2 = opt->l2;
    const int sr = opt->stochastic && rng;
    tk_rng r = sr ? *rng : (tk_rng){0};                 /* keep RNG state in a register */
    int i = 0;
#if defined(__aarch64__) && TK_DTYPE == TK_DTYPE_BFLOAT16
    /* Plain-SGD bf16 fast path: the scalar loop pays a tk_float_to_bf16 call
     * per weight (the requantizing store dominates, not the FMA). Widen with a
     * shift, update with a fused vfmsq -- the same single-rounding w - lr*g the
     * scalar fmsub computes -- and narrow in-register: bit-identical
     * trajectories, no call. L1/L2/SR take the scalar loop below. */
    if (!sr && l1 == 0.0f && l2 == 0.0f) {
        const float32x4_t vlr = vdupq_n_f32(lr);
        for (; i + 8 <= n; i += 8) {
            const uint16x8_t h = vld1q_u16(W + i);
            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
            w0 = vfmsq_f32(w0, vlr, vld1q_f32(dW + i));
            w1 = vfmsq_f32(w1, vlr, vld1q_f32(dW + i + 4));
            vst1q_u16(W + i, vcombine_u16(tk_bf16_rne_narrow(w0), tk_bf16_rne_narrow(w1)));
        }
    }
#endif
    for (; i < n; i++) {
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
