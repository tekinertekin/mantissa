#include "ops.h"
#include "pool.h"
#include <stdlib.h>

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

#if defined(__clang__)
#define TK_HAVE_BF16_MLAL 1
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

/* FEAT_BF16 (ARMv8.6, Apple M2+, Graviton3+): cached runtime check, same
 * pattern as the AVX2 dispatch — compile unconditionally via the target
 * attribute, call only when the CPU reports the feature. Clang-only: GCC's
 * arm_neon.h gates the bf16 intrinsics on global -march. */
static int tk__cpu_has_bf16(void) {
    static int cached = -1;
    if (cached < 0) {
#if defined(__APPLE__)
        int v = 0; size_t sz = sizeof v;
        cached = (sysctlbyname("hw.optional.arm.FEAT_BF16", &v, &sz, NULL, 0) == 0 && v != 0);
#elif defined(__linux__) && defined(HWCAP2_BF16)
        cached = (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#else
        cached = 0;
#endif
    }
    return cached;
}

/* BFMLALB/BFMLALT: exact f32 FMAs of exact bf16 products, 2 instructions per
 * row per 8 elements (vs 4 widen+FMA). Two accumulator chains per row — a
 * single chain is FMA-latency bound and measures no faster than widen+FMA;
 * split even/odd chains measure ~1.9-2.4x over the depth-4 kernel (M4). */
__attribute__((target("arch=armv8.6-a+bf16")))
static void tk__dot4_bfmlal(const tk_bf16_t *W, int in, const tk_bf16_t *x, float *out) {
    const tk_bf16_t *w0 = W, *w1 = W + in, *w2 = W + 2 * in, *w3 = W + 3 * in;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    int i = 0;
    for (; i + 8 <= in; i += 8) {
        bfloat16x8_t xv = vreinterpretq_bf16_u16(vld1q_u16(x + i));
        bfloat16x8_t v0 = vreinterpretq_bf16_u16(vld1q_u16(w0 + i));
        bfloat16x8_t v1 = vreinterpretq_bf16_u16(vld1q_u16(w1 + i));
        bfloat16x8_t v2 = vreinterpretq_bf16_u16(vld1q_u16(w2 + i));
        bfloat16x8_t v3 = vreinterpretq_bf16_u16(vld1q_u16(w3 + i));
        a0 = vbfmlalbq_f32(a0, v0, xv); b0 = vbfmlaltq_f32(b0, v0, xv);
        a1 = vbfmlalbq_f32(a1, v1, xv); b1 = vbfmlaltq_f32(b1, v1, xv);
        a2 = vbfmlalbq_f32(a2, v2, xv); b2 = vbfmlaltq_f32(b2, v2, xv);
        a3 = vbfmlalbq_f32(a3, v3, xv); b3 = vbfmlaltq_f32(b3, v3, xv);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0, b0)), s1 = vaddvq_f32(vaddq_f32(a1, b1));
    float s2 = vaddvq_f32(vaddq_f32(a2, b2)), s3 = vaddvq_f32(vaddq_f32(a3, b3));
    for (; i < in; i++) {
        float xi = tk_bf16_to_float(x[i]);
        s0 += tk_bf16_to_float(w0[i]) * xi; s1 += tk_bf16_to_float(w1[i]) * xi;
        s2 += tk_bf16_to_float(w2[i]) * xi; s3 += tk_bf16_to_float(w3[i]) * xi;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}
#endif /* __clang__ */
#elif TK_DTYPE == TK_DTYPE_FP16
#define TK_HAVE_NEON_FP16 1
/* Load 4 fp16 as float32: a single FCVTL (base A64) per vector — IEEE-exact,
 * including subnormals, same contract as the scalar FCVT read path. */
static inline float32x4_t tk__fp16x4(const tk_fp16_t *p) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
}
#endif

#elif defined(__x86_64__) && (TK_DTYPE == TK_DTYPE_FLOAT32 || TK_DTYPE == TK_DTYPE_BFLOAT16)
#include <immintrin.h>
#define TK_HAVE_AVX2 1

/* Runtime CPU dispatch: the AVX2 kernel is compiled unconditionally (via the
 * per-function target attribute) but only *called* when the running CPU has
 * AVX2+FMA, so one portable binary is fast on modern x86 and correct on old
 * chips (and under emulators without AVX). NEON on arm64 needs none of this —
 * it is baseline. */
static int tk__cpu_has_avx2(void) {
    static int cached = -1;
    if (cached < 0) cached = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));
    return cached;
}

__attribute__((target("avx2,fma")))
static float tk__hsum256(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

#if TK_DTYPE == TK_DTYPE_BFLOAT16
/* Load 8 bfloat16 as float32: zero-extend u16->u32 and shift into the high half. */
__attribute__((target("avx2,fma")))
static __m256 tk__bf16x8(const tk_bf16_t *p) {
    __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16);
    return _mm256_castsi256_ps(w);
}
#endif

/* Four dot products, 8-wide AVX2 FMA accumulators (one per weight row). */
__attribute__((target("avx2,fma")))
static void tk__dot4_avx2(const tk_scalar_t *W, int in, const tk_scalar_t *x, float *out) {
    const tk_scalar_t *w0 = W, *w1 = W + in, *w2 = W + 2 * in, *w3 = W + 3 * in;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    int i = 0;
    for (; i + 8 <= in; i += 8) {
#if TK_DTYPE == TK_DTYPE_FLOAT32
        __m256 xv = _mm256_loadu_ps(x + i);
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + i), xv, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + i), xv, a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + i), xv, a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + i), xv, a3);
#else
        __m256 xv = tk__bf16x8(x + i);
        a0 = _mm256_fmadd_ps(tk__bf16x8(w0 + i), xv, a0);
        a1 = _mm256_fmadd_ps(tk__bf16x8(w1 + i), xv, a1);
        a2 = _mm256_fmadd_ps(tk__bf16x8(w2 + i), xv, a2);
        a3 = _mm256_fmadd_ps(tk__bf16x8(w3 + i), xv, a3);
#endif
    }
    float s0 = tk__hsum256(a0), s1 = tk__hsum256(a1);
    float s2 = tk__hsum256(a2), s3 = tk__hsum256(a3);
    for (; i < in; i++) {
        float xi = TK_TO_FLOAT(x[i]);
        s0 += TK_TO_FLOAT(w0[i]) * xi; s1 += TK_TO_FLOAT(w1[i]) * xi;
        s2 += TK_TO_FLOAT(w2[i]) * xi; s3 += TK_TO_FLOAT(w3[i]) * xi;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}
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
 * Register blocking: each x element is loaded once and feeds independent FMA
 * chains, so the FP units stay busy and x conversions are shared across rows —
 * a single-row loop can do neither. On arm64, float32/bfloat16/fp16 use
 * explicit NEON kernels with two accumulator chains per row (depth-8); bf16
 * additionally dispatches to a BFMLALB/T kernel on FEAT_BF16 CPUs. x86-64
 * runtime-dispatches AVX2+FMA for float32/bfloat16. */
static inline void tk__dot4(const tk_scalar_t *restrict W, int in,
                            const tk_scalar_t *restrict x,
                            float *restrict out) {
    const tk_scalar_t *restrict w0 = W, *restrict w1 = W + in,
                       *restrict w2 = W + 2 * in, *restrict w3 = W + 3 * in;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
#if defined(TK_HAVE_NEON_F32) || defined(TK_HAVE_NEON_BF16) || defined(TK_HAVE_NEON_FP16)
  #if defined(TK_HAVE_BF16_MLAL)
    if (tk__cpu_has_bf16()) { tk__dot4_bfmlal(W, in, x, out); return; }
  #endif
  #if defined(TK_HAVE_NEON_F32)
    #define TK__LD4(p) vld1q_f32(p)
  #elif defined(TK_HAVE_NEON_BF16)
    #define TK__LD4(p) tk__bf16x4(p)         /* load + widen */
  #else
    #define TK__LD4(p) tk__fp16x4(p)         /* load + FCVTL */
  #endif
    /* Two accumulator chains per row (depth-8): one chain is bound by FMA
     * latency, not throughput — splitting it measured ~1.6-2x (M4, bf16). */
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (; i + 8 <= in; i += 8) {
        float32x4_t x0 = TK__LD4(x + i), x1 = TK__LD4(x + i + 4);
        a0 = vfmaq_f32(a0, TK__LD4(w0 + i), x0); b0 = vfmaq_f32(b0, TK__LD4(w0 + i + 4), x1);
        a1 = vfmaq_f32(a1, TK__LD4(w1 + i), x0); b1 = vfmaq_f32(b1, TK__LD4(w1 + i + 4), x1);
        a2 = vfmaq_f32(a2, TK__LD4(w2 + i), x0); b2 = vfmaq_f32(b2, TK__LD4(w2 + i + 4), x1);
        a3 = vfmaq_f32(a3, TK__LD4(w3 + i), x0); b3 = vfmaq_f32(b3, TK__LD4(w3 + i + 4), x1);
    }
  #undef TK__LD4
    s0 = vaddvq_f32(vaddq_f32(a0, b0)); s1 = vaddvq_f32(vaddq_f32(a1, b1));
    s2 = vaddvq_f32(vaddq_f32(a2, b2)); s3 = vaddvq_f32(vaddq_f32(a3, b3));
#elif defined(TK_HAVE_AVX2)
    if (tk__cpu_has_avx2()) { tk__dot4_avx2(W, in, x, out); return; }  /* else portable below */
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

typedef struct {
    const tk_scalar_t *W, *x, *bias;
    float *y;
    int in_dim;
    tk_activation_t act;
} tk_gemv_ctx;

static void gemv_worker(void *p, int o0, int o1, int worker) {
    (void)worker;
    const tk_gemv_ctx *c = (const tk_gemv_ctx *)p;
    gemv_range(c->W, c->x, c->bias, c->y, o0, o1, c->in_dim, c->act);
}

void tk_linear_forward(const tk_scalar_t *restrict W,
                       const tk_scalar_t *restrict x,
                       const tk_scalar_t *restrict bias,
                       float *restrict y,
                       int out_dim, int in_dim,
                       tk_activation_t act) {
    if ((size_t)out_dim * in_dim >= TK_MT_MIN_WORK && tk_num_threads() > 1) {
        tk_gemv_ctx c = { W, x, bias, y, in_dim, act };
        tk_parallel_for(out_dim, gemv_worker, &c);   /* split rows across the pool */
    } else {
        gemv_range(W, x, bias, y, 0, out_dim, in_dim, act);
    }
}

/* Batch forward (GEMM): rows [o0,o1) against ALL samples. Sample-inner order
 * keeps the current 4-row W block hot in L1 across the sweep, so W streams
 * from memory once per batch instead of once per sample — the GEMV loop's
 * bandwidth bound amortized over n. Same tk__dot4 kernel; per-row numerics
 * match the single-sample path. */
static void gemm_range(const tk_scalar_t *restrict W, const tk_scalar_t *restrict X,
                       const tk_scalar_t *restrict bias, float *restrict Y,
                       int o0, int o1, int n_samples, int out_dim, int in_dim,
                       tk_activation_t act) {
    int o = o0;
    for (; o + 4 <= o1; o += 4) {
        for (int s = 0; s < n_samples; s++) {
            float r[4];
            tk__dot4(W + (size_t)o * in_dim, in_dim, X + (size_t)s * in_dim, r);
            float *ys = Y + (size_t)s * out_dim + o;
            for (int k = 0; k < 4; k++) {
                float z = r[k] + (bias ? TK_TO_FLOAT(bias[o + k]) : 0.0f);
                ys[k] = tk_act_scalar(z, act);
            }
        }
    }
    for (; o < o1; o++) {
        for (int s = 0; s < n_samples; s++) {
            float z = tk_dot(W + (size_t)o * in_dim, X + (size_t)s * in_dim, in_dim);
            if (bias) z += TK_TO_FLOAT(bias[o]);
            Y[(size_t)s * out_dim + o] = tk_act_scalar(z, act);
        }
    }
}

typedef struct {
    const tk_scalar_t *W, *X, *bias;
    float *Y;
    int n_samples, out_dim, in_dim;
    tk_activation_t act;
} tk_gemm_ctx;

static void gemm_worker(void *p, int o0, int o1, int worker) {
    (void)worker;
    const tk_gemm_ctx *c = (const tk_gemm_ctx *)p;
    gemm_range(c->W, c->X, c->bias, c->Y, o0, o1,
               c->n_samples, c->out_dim, c->in_dim, c->act);
}

void tk_linear_forward_batch(const tk_scalar_t *restrict W,
                             const tk_scalar_t *restrict X,
                             const tk_scalar_t *restrict bias,
                             float *restrict Y,
                             int n_samples, int out_dim, int in_dim,
                             tk_activation_t act) {
    if ((size_t)out_dim * in_dim * (size_t)(n_samples > 0) >= TK_MT_MIN_WORK
        && tk_num_threads() > 1) {
        tk_gemm_ctx c = { W, X, bias, Y, n_samples, out_dim, in_dim, act };
        tk_parallel_for(out_dim, gemm_worker, &c);   /* threads own row ranges */
        return;
    }
    gemm_range(W, X, bias, Y, 0, out_dim, n_samples, out_dim, in_dim, act);
}

/* Quantize a float through the active storage type (round-trip), so the f32
 * API reproduces that type's precision on plain-float inputs. */
static inline float tk_q(float v) { return TK_TO_FLOAT(TK_FROM_FLOAT(v)); }

void tk_quantize(const float *restrict src, tk_scalar_t *restrict dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = TK_FROM_FLOAT(src[i]);
}

/* Serial f32-API forward. Kept in its own function, deliberately free of the
 * pool-dispatch call: routing the multithread branch's W/x/y pointers into an
 * external pool worker in the same function makes them escape, which defeats
 * the compiler's no-alias analysis and de-vectorizes THIS loop (~40% slower,
 * measured). Isolating the serial loop here restores its auto-vectorized
 * codegen while the wrapper below still gets to thread.
 *
 * Quantizes x once up front (x is read out_dim times, so this removes the
 * x-side round-trip from the hot loop); W is quantized inline because it
 * arrives as float. Falls back to inline x-quantization if scratch can't be
 * allocated. */
static void tk_linear_forward_f32_serial(const float *restrict W,
                                         const float *restrict x,
                                         const float *restrict bias,
                                         float *restrict y,
                                         int out_dim, int in_dim,
                                         tk_activation_t act) {
    float *xq = (float *)malloc((size_t)in_dim * sizeof(float));
    if (xq) for (int i = 0; i < in_dim; i++) xq[i] = tk_q(x[i]);

    for (int o = 0; o < out_dim; o++) {
        const float *restrict wr = W + (size_t)o * in_dim;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        if (xq) {
            for (; i + 4 <= in_dim; i += 4) {
                s0 += tk_q(wr[i + 0]) * xq[i + 0];
                s1 += tk_q(wr[i + 1]) * xq[i + 1];
                s2 += tk_q(wr[i + 2]) * xq[i + 2];
                s3 += tk_q(wr[i + 3]) * xq[i + 3];
            }
            for (; i < in_dim; i++) s0 += tk_q(wr[i]) * xq[i];
        } else {
            for (; i < in_dim; i++) s0 += tk_q(wr[i]) * tk_q(x[i]);
        }
        float z = (s0 + s1) + (s2 + s3);
        if (bias) z += tk_q(bias[o]);
        y[o] = tk_act_scalar(z, act);
    }
    free(xq);
}

/* Output rows [o0, o1) for the pooled path; xq is pre-quantized and shared
 * read-only across workers, W is quantized inline. Rows are disjoint. */
static void gemv_f32_range(const float *restrict W, const float *restrict xq,
                           const float *restrict bias, float *restrict y,
                           int o0, int o1, int in_dim, tk_activation_t act) {
    for (int o = o0; o < o1; o++) {
        const float *restrict wr = W + (size_t)o * in_dim;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= in_dim; i += 4) {
            s0 += tk_q(wr[i + 0]) * xq[i + 0];
            s1 += tk_q(wr[i + 1]) * xq[i + 1];
            s2 += tk_q(wr[i + 2]) * xq[i + 2];
            s3 += tk_q(wr[i + 3]) * xq[i + 3];
        }
        for (; i < in_dim; i++) s0 += tk_q(wr[i]) * xq[i];
        float z = (s0 + s1) + (s2 + s3);
        if (bias) z += tk_q(bias[o]);
        y[o] = tk_act_scalar(z, act);
    }
}

typedef struct {
    const float *W, *xq, *bias;
    float *y;
    int in_dim;
    tk_activation_t act;
} tk_gemv_f32_ctx;

static void gemv_f32_worker(void *p, int o0, int o1, int worker) {
    (void)worker;
    const tk_gemv_f32_ctx *c = (const tk_gemv_f32_ctx *)p;
    gemv_f32_range(c->W, c->xq, c->bias, c->y, o0, o1, c->in_dim, c->act);
}

void tk_linear_forward_f32(const float *restrict W,
                           const float *restrict x,
                           const float *restrict bias,
                           float *restrict y,
                           int out_dim, int in_dim,
                           tk_activation_t act) {
    /* Rows are independent (disjoint y[o]), so threading changes no per-row
     * numerics -- results are bit-identical to the serial path. Threshold and
     * dispatch mirror tk_linear_forward; on the serial/small-layer path we call
     * the isolated serial routine to keep its auto-vectorized codegen. */
    if ((size_t)out_dim * in_dim >= TK_MT_MIN_WORK && tk_num_threads() > 1) {
        float *xq = (float *)malloc((size_t)in_dim * sizeof(float));
        if (xq) {
            for (int i = 0; i < in_dim; i++) xq[i] = tk_q(x[i]);
            tk_gemv_f32_ctx c = { W, xq, bias, y, in_dim, act };
            tk_parallel_for(out_dim, gemv_f32_worker, &c);
            free(xq);
            return;
        }
    }
    tk_linear_forward_f32_serial(W, x, bias, y, out_dim, in_dim, act);
}
