#include "ops.h"
#include "pool.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#if TK_DTYPE == TK_DTYPE_FLOAT32
#define TK_HAVE_NEON_F32 1
#elif TK_DTYPE == TK_DTYPE_BFLOAT16
#define TK_HAVE_NEON_BF16 1
/* Load 4 bfloat16 as float32: widen u16->u32 and shift the bits into the high
 * half — bf16 is exactly the top 16 bits of a float32, so this is the conversion. */
static inline float32x4_t tk__bf16x4(const tk_bf16_t *p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}
#endif
#endif

float tk_dot(const tk_scalar_t *restrict a, const tk_scalar_t *restrict b, int n) {
    /* Four independent accumulators break the loop-carried dependency on the FP
     * adder, letting the CPU pipeline/vectorize the FMAs instead of stalling on
     * each add's latency. The compiler folds mul+add into FMA under
     * -ffp-contract=fast. */
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += TK_TO_FLOAT(a[i + 0]) * TK_TO_FLOAT(b[i + 0]);
        s1 += TK_TO_FLOAT(a[i + 1]) * TK_TO_FLOAT(b[i + 1]);
        s2 += TK_TO_FLOAT(a[i + 2]) * TK_TO_FLOAT(b[i + 2]);
        s3 += TK_TO_FLOAT(a[i + 3]) * TK_TO_FLOAT(b[i + 3]);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += TK_TO_FLOAT(a[i]) * TK_TO_FLOAT(b[i]);
    return s;
}

/* Four dot products (4 consecutive weight rows against the same x) at once.
 * Register blocking: each x element is loaded once and feeds 4 independent FMA
 * chains, so the FP units stay busy and x conversions are shared across rows —
 * a single-row loop can do neither. ~1.3x portable; the float32 build adds an
 * explicit NEON FMA kernel on arm64 for ~2x. */
static inline void tk__dot4(const tk_scalar_t *restrict W, int in,
                            const tk_scalar_t *restrict x,
                            float *restrict out) {
    const tk_scalar_t *restrict w0 = W, *restrict w1 = W + in,
                       *restrict w2 = W + 2 * in, *restrict w3 = W + 3 * in;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
#if defined(TK_HAVE_NEON_F32) || defined(TK_HAVE_NEON_BF16)
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (; i + 4 <= in; i += 4) {
  #ifdef TK_HAVE_NEON_F32
        float32x4_t xv = vld1q_f32(x + i);
        a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), xv);
        a1 = vfmaq_f32(a1, vld1q_f32(w1 + i), xv);
        a2 = vfmaq_f32(a2, vld1q_f32(w2 + i), xv);
        a3 = vfmaq_f32(a3, vld1q_f32(w3 + i), xv);
  #else /* bf16: load + widen each vector */
        float32x4_t xv = tk__bf16x4(x + i);
        a0 = vfmaq_f32(a0, tk__bf16x4(w0 + i), xv);
        a1 = vfmaq_f32(a1, tk__bf16x4(w1 + i), xv);
        a2 = vfmaq_f32(a2, tk__bf16x4(w2 + i), xv);
        a3 = vfmaq_f32(a3, tk__bf16x4(w3 + i), xv);
  #endif
    }
    s0 = vaddvq_f32(a0); s1 = vaddvq_f32(a1); s2 = vaddvq_f32(a2); s3 = vaddvq_f32(a3);
#endif
    for (; i < in; i++) {
        float xi = TK_TO_FLOAT(x[i]);
        s0 += TK_TO_FLOAT(w0[i]) * xi; s1 += TK_TO_FLOAT(w1[i]) * xi;
        s2 += TK_TO_FLOAT(w2[i]) * xi; s3 += TK_TO_FLOAT(w3[i]) * xi;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

/* Compute output rows [o0, o1) — the unit of work a thread owns (rows are
 * disjoint, so no locking). Uses the 4-row register-blocked kernel internally. */
static void gemv_range(const tk_scalar_t *restrict W, const tk_scalar_t *restrict x,
                       const tk_scalar_t *restrict bias, float *restrict y,
                       int o0, int o1, int in_dim, tk_activation_t act) {
    int o = o0;
    for (; o + 4 <= o1; o += 4) {
        float s[4];
        tk__dot4(W + (size_t)o * in_dim, in_dim, x, s);
        for (int k = 0; k < 4; k++) {
            float z = s[k] + (bias ? TK_TO_FLOAT(bias[o + k]) : 0.0f);
            y[o + k] = tk_act_scalar(z, act);
        }
    }
    for (; o < o1; o++) {
        float z = tk_dot(W + (size_t)o * in_dim, x, in_dim);
        if (bias) z += TK_TO_FLOAT(bias[o]);
        y[o] = tk_act_scalar(z, act);
    }
}

/* Parallelize only above this much work (out_dim * in_dim); small layers run
 * serially so the pool overhead never hurts the millions-of-small-calls case. */
#define TK_MT_MIN_WORK (1 << 18)

typedef struct {
    const tk_scalar_t *W, *x, *bias;
    float *y;
    int in_dim;
    tk_activation_t act;
} tk_gemv_ctx;

static void gemv_worker(void *p, int o0, int o1) {
    const tk_gemv_ctx *c = (const tk_gemv_ctx *)p;
    gemv_range(c->W, c->x, c->bias, c->y, o0, o1, c->in_dim, c->act);
}

void tk_linear_forward(const tk_scalar_t *restrict W,
                       const tk_scalar_t *restrict x,
                       const tk_scalar_t *restrict bias,
                       float *restrict y,
                       int out_dim, int in_dim,
                       tk_activation_t act) {
    if ((long)out_dim * in_dim >= TK_MT_MIN_WORK && tk_num_threads() > 1) {
        tk_gemv_ctx c = { W, x, bias, y, in_dim, act };
        tk_parallel_for(out_dim, gemv_worker, &c);   /* split rows across the pool */
    } else {
        gemv_range(W, x, bias, y, 0, out_dim, in_dim, act);
    }
}

/* Quantize a float through the active storage type (round-trip), so the f32
 * API reproduces that type's precision on plain-float inputs. */
static inline float tk_q(float v) { return TK_TO_FLOAT(TK_FROM_FLOAT(v)); }

void tk_linear_forward_f32(const float *restrict W,
                           const float *restrict x,
                           const float *restrict bias,
                           float *restrict y,
                           int out_dim, int in_dim,
                           tk_activation_t act) {
    for (int o = 0; o < out_dim; o++) {
        const float *restrict wr = W + (size_t)o * in_dim;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= in_dim; i += 4) {
            s0 += tk_q(wr[i + 0]) * tk_q(x[i + 0]);
            s1 += tk_q(wr[i + 1]) * tk_q(x[i + 1]);
            s2 += tk_q(wr[i + 2]) * tk_q(x[i + 2]);
            s3 += tk_q(wr[i + 3]) * tk_q(x[i + 3]);
        }
        float z = (s0 + s1) + (s2 + s3);
        for (; i < in_dim; i++) z += tk_q(wr[i]) * tk_q(x[i]);
        if (bias) z += tk_q(bias[o]);
        y[o] = tk_act_scalar(z, act);
    }
}
