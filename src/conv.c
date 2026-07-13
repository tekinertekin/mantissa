#include "conv.h"
#include "pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__aarch64__)
#include <arm_neon.h>   /* f32 dot4 kernel in tk__dot4_f32 */
#endif

/* CNN primitives, pure float32 (no storage-dtype quantization on this path).
 *
 * Convolution is im2col + GEMM (Chellapilla et al., 2006, "High Performance
 * Convolutional Neural Networks for Document Processing"): each sample's
 * input is unrolled into a patch matrix so the conv becomes the dense-layer
 * dot-product sweep ops.c is built around. The patch matrix is heap scratch
 * allocated ONCE per call and reused across the batch loop -- materializing
 * im2col for the whole batch at once would multiply the footprint by n for
 * no speed win. Threading is across batch samples (one scratch slice per
 * worker), gated on TK_MT_MIN_WORK like ops.c; backward's batch-summed dK/db
 * use per-worker partials, the tk_linear_backward reduction pattern. If
 * scratch allocation fails, direct (no-scratch) loops keep the result exact. */

int tk_conv2d_out_dim(int in, int k, int stride, int pad) {
    if (in <= 0 || k <= 0 || stride <= 0 || pad < 0) return 0;
    const int out = (in + 2 * pad - k) / stride + 1;
    return out > 0 ? out : 0;
}

/* Unroll one sample into patch rows: cols is (oh*ow) x (in_c*kh*kw),
 * row-major, out-of-image taps zero-filled. */
static void im2col(const float *restrict x, float *restrict cols,
                   int in_c, int in_h, int in_w, int kh, int kw,
                   int stride, int pad, int oh, int ow) {
    float *restrict col = cols;
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            const int iy0 = oy * stride - pad, ix0 = ox * stride - pad;
            for (int c = 0; c < in_c; c++) {
                const float *restrict plane = x + (size_t)c * in_h * in_w;
                for (int ky = 0; ky < kh; ky++) {
                    const int iy = iy0 + ky;
                    if (iy < 0 || iy >= in_h) {
                        for (int kx = 0; kx < kw; kx++) *col++ = 0.0f;
                        continue;
                    }
                    const float *restrict row = plane + (size_t)iy * in_w;
                    for (int kx = 0; kx < kw; kx++) {
                        const int ix = ix0 + kx;
                        *col++ = (ix >= 0 && ix < in_w) ? row[ix] : 0.0f;
                    }
                }
            }
        }
    }
}

/* Four f32 dot products: 4 contiguous rows of W against one x, so each x
 * load feeds four independent FMA chains -- the tk__dot4 register-blocking
 * recipe from ops.c, f32-only (this family never quantizes). NEON is arm64
 * baseline; two accumulator chains per row cover FMA latency. Other targets
 * take the portable loop. Measured (M4, VGG-block GEMM, 64 x 1024 x 576):
 * 8.5 GFLOP/s single-row scalar -> 13.8 pure-C 4-row -> 63.4 with NEON. */
static void tk__dot4_f32(const float *restrict W, int in,
                         const float *restrict x, float *restrict out) {
    const float *restrict w0 = W, *restrict w1 = W + in,
                *restrict w2 = W + 2 * in, *restrict w3 = W + 3 * in;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
#if defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    for (; i + 8 <= in; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        a0 = vfmaq_f32(a0, vld1q_f32(w0 + i), x0); b0 = vfmaq_f32(b0, vld1q_f32(w0 + i + 4), x1);
        a1 = vfmaq_f32(a1, vld1q_f32(w1 + i), x0); b1 = vfmaq_f32(b1, vld1q_f32(w1 + i + 4), x1);
        a2 = vfmaq_f32(a2, vld1q_f32(w2 + i), x0); b2 = vfmaq_f32(b2, vld1q_f32(w2 + i + 4), x1);
        a3 = vfmaq_f32(a3, vld1q_f32(w3 + i), x0); b3 = vfmaq_f32(b3, vld1q_f32(w3 + i + 4), x1);
    }
    s0 = vaddvq_f32(vaddq_f32(a0, b0)); s1 = vaddvq_f32(vaddq_f32(a1, b1));
    s2 = vaddvq_f32(vaddq_f32(a2, b2)); s3 = vaddvq_f32(vaddq_f32(a3, b3));
#endif
    for (; i < in; i++) {
        const float xi = x[i];
        s0 += w0[i] * xi; s1 += w1[i] * xi;
        s2 += w2[i] * xi; s3 += w3[i] * xi;
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

/* Single-row f32 dot (block tail); four accumulators, same as tk_dot. */
static float tk__dot1_f32(const float *restrict w, const float *restrict x,
                          int in) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 4 <= in; i += 4) {
        s0 += w[i + 0] * x[i + 0];
        s1 += w[i + 1] * x[i + 1];
        s2 += w[i + 2] * x[i + 2];
        s3 += w[i + 3] * x[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < in; i++) s += w[i] * x[i];
    return s;
}

/* One sample's forward GEMM: Z/Y rows are out_c x npatch, 4 filter rows per
 * sweep so every patch load is shared (register blocking, see tk__dot4_f32). */
static void conv_gemm(const float *restrict K, const float *restrict cols,
                      const float *restrict bias,
                      float *restrict Zs, float *restrict Ys,
                      int out_c, int npatch, int kdim, tk_activation_t act) {
    int oc = 0;
    for (; oc + 4 <= out_c; oc += 4) {
        for (int p = 0; p < npatch; p++) {
            float s[4];
            tk__dot4_f32(K + (size_t)oc * kdim, kdim,
                         cols + (size_t)p * kdim, s);
            for (int k = 0; k < 4; k++) {
                const float z = s[k] + (bias ? bias[oc + k] : 0.0f);
                if (Zs) Zs[(size_t)(oc + k) * npatch + p] = z;
                Ys[(size_t)(oc + k) * npatch + p] = tk_act_scalar(z, act);
            }
        }
    }
    for (; oc < out_c; oc++) {
        const float *restrict kr = K + (size_t)oc * kdim;
        float *restrict zr = Zs ? Zs + (size_t)oc * npatch : NULL;
        float *restrict yr = Ys + (size_t)oc * npatch;
        const float b = bias ? bias[oc] : 0.0f;
        for (int p = 0; p < npatch; p++) {
            const float z = b + tk__dot1_f32(kr, cols + (size_t)p * kdim, kdim);
            if (zr) zr[p] = z;
            yr[p] = tk_act_scalar(z, act);
        }
    }
}

/* Scratch-free fallback (and the reference the tests cross-check): direct
 * bounds-checked loops over samples [s0, s1). */
static void conv_fwd_direct(const float *restrict X, const float *restrict K,
                            const float *restrict bias,
                            float *restrict Z, float *restrict Y,
                            int s0, int s1, int in_c, int in_h, int in_w,
                            int out_c, int kh, int kw, int stride, int pad,
                            int oh, int ow, tk_activation_t act) {
    const size_t in_sz = (size_t)in_c * in_h * in_w;
    const size_t out_sz = (size_t)out_c * oh * ow;
    for (int s = s0; s < s1; s++) {
        const float *restrict xs = X + (size_t)s * in_sz;
        for (int oc = 0; oc < out_c; oc++) {
            const float *restrict kf = K + (size_t)oc * in_c * kh * kw;
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    float z = bias ? bias[oc] : 0.0f;
                    const int iy0 = oy * stride - pad, ix0 = ox * stride - pad;
                    for (int c = 0; c < in_c; c++)
                        for (int ky = 0; ky < kh; ky++) {
                            const int iy = iy0 + ky;
                            if (iy < 0 || iy >= in_h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                const int ix = ix0 + kx;
                                if (ix < 0 || ix >= in_w) continue;
                                z += kf[((size_t)c * kh + ky) * kw + kx]
                                   * xs[((size_t)c * in_h + iy) * in_w + ix];
                            }
                        }
                    const size_t o = (size_t)s * out_sz
                                   + ((size_t)oc * oh + oy) * ow + ox;
                    if (Z) Z[o] = z;
                    Y[o] = tk_act_scalar(z, act);
                }
            }
        }
    }
}

/* --- batched conv GEMM (large shapes) --------------------------------------
 *
 * The per-sample path above re-streams K (out_c x kdim) from L2 for every
 * sample and re-reads the sample's im2col scratch once per 4-row filter
 * block. For large shapes the whole batch is instead ONE GEMM,
 *   C (out_c x n*oh*ow) = A (K, out_c x kdim) . B (im2col, kdim x n*oh*ow),
 * blocked BLIS-style (Van Zee & van de Geijn, 2015, "BLIS: A Framework for
 * Rapidly Instantiating BLAS Functionality"): A is packed once per call into
 * MR-row micro-panels, B is packed panel-by-panel (NC columns at a time, a
 * panel spans batch samples) into NR-column micro-panels, and an MR x NR
 * register tile of C accumulates over the full kdim. The full-batch im2col
 * matrix is never materialized (rejected in RELEASES v0.2.1: n x the
 * footprint); per-worker footprint is one kdim x NC panel.
 *
 * kdim is NOT blocked: target shapes keep it small (<= 576 for a 64-channel
 * 3x3 layer), so one A micro-panel (MR*kdim) plus one B micro-panel
 * (kdim*NR) sit comfortably in the M4's 128 KB L1d and the register tile
 * accumulates the whole dot in one pass -- no partial-C traffic, and bias +
 * activation fold into the tile store. Threading is over column panels;
 * every output element is accumulated in the same k-order by the same
 * kernel, so the forward is bit-identical across thread counts. */

#if defined(__aarch64__)
  /* 6x16 f32 tile: 24 accumulator registers + 1 A + 4 B of the 32 NEON
   * registers. An 8x12 tile (same 24 accumulators) measured ~8% slower once
   * packing became run-based: NR=16 needs fewer micro-panels (less pack
   * bookkeeping, and 16 | 32 = the common ow, so runs stay whole) and
   * streams less A per FLOP. See DESIGN.md, rejected list;
   * TK_CONV_TILE_8X12 keeps the loser buildable for re-measurement. */
  #if defined(TK_CONV_TILE_8X12)
enum { TK_CONV_MR = 8, TK_CONV_NR = 12 };
  #else
enum { TK_CONV_MR = 6, TK_CONV_NR = 16 };
  #endif
#elif defined(__x86_64__)
/* 6x16: 12 YMM accumulators + 2 B + 1 A broadcast of the 16 YMM registers. */
enum { TK_CONV_MR = 6, TK_CONV_NR = 16 };
#else
enum { TK_CONV_MR = 4, TK_CONV_NR = 4 };
#endif

/* Columns per packed B panel (the per-worker working set is kdim*NC floats).
 * Swept 128/256/512 on M4 (see DESIGN.md); flat within noise, 256 kept. */
#ifndef TK_CONV_NC
#define TK_CONV_NC 256
#endif

/* MACs below this run the per-sample path: its scratch is smaller and the
 * GEMM's pack/tile overhead only amortizes on real work. Swept on M4: the
 * LeNet rows (2.8M/7.7M MACs) are faster on the per-sample path, the VGG
 * rows (28M+) on the GEMM path; 2^24 splits them with margin. */
#define TK_CONV_GEMM_MIN_MACS ((size_t)1 << 24)

typedef struct {
    int n, in_c, in_h, in_w, out_c, kh, kw, stride, pad, oh, ow;
    int npatch, kdim;
    size_t in_sz, out_sz;        /* per-sample X / Y sizes */
    long ncols;                  /* n * npatch, the GEMM N dimension */
} tk_conv_geom;

static tk_conv_geom conv_geom(int n, int in_c, int in_h, int in_w, int out_c,
                              int kh, int kw, int stride, int pad,
                              int oh, int ow) {
    tk_conv_geom g;
    g.n = n; g.in_c = in_c; g.in_h = in_h; g.in_w = in_w; g.out_c = out_c;
    g.kh = kh; g.kw = kw; g.stride = stride; g.pad = pad; g.oh = oh; g.ow = ow;
    g.npatch = oh * ow; g.kdim = in_c * kh * kw;
    g.in_sz = (size_t)in_c * in_h * in_w;
    g.out_sz = (size_t)out_c * g.npatch;
    g.ncols = (long)n * g.npatch;
    return g;
}

/* Pack the filter matrix into MR-row micro-panels: Ap[panel][k][MR],
 * k-major so the kernel loads MR contiguous A values per k step; rows past
 * out_c are zero-filled so edge tiles run the same kernel. */
static void convg_pack_A(const float *restrict K, float *restrict Ap,
                         int out_c, int kdim) {
    const int MR = TK_CONV_MR;
    for (int oc0 = 0; oc0 < out_c; oc0 += MR) {
        const int m = (out_c - oc0 < MR) ? out_c - oc0 : MR;
        float *restrict dst = Ap + (size_t)(oc0 / MR) * kdim * MR;
        for (int k = 0; k < kdim; k++)
            for (int i = 0; i < MR; i++)
                *dst++ = (i < m) ? K[(size_t)(oc0 + i) * kdim + k] : 0.0f;
    }
}

/* Pack `run` consecutive-ox columns (one sample, one output row) into a
 * micro-panel at column offset `dst`: dst[k*NR + t], k walks (c, ky, kx),
 * out-of-image taps zero-filled. At stride 1 the t loop is a contiguous
 * clipped copy from one X row -- the packing cost is a straight stream, not
 * a per-element bounds check (measured 1.5x on the whole forward). */
static void convg_pack_run(const float *restrict xs, float *restrict dst,
                           int run, int oy, int ox, const tk_conv_geom *g) {
    const int NR = TK_CONV_NR;
    const int iy0 = oy * g->stride - g->pad, ixb = ox * g->stride - g->pad;
    float *restrict d = dst;
    for (int c = 0; c < g->in_c; c++) {
        const float *restrict plane = xs + (size_t)c * g->in_h * g->in_w;
        for (int ky = 0; ky < g->kh; ky++) {
            const int iy = iy0 + ky;
            if (iy < 0 || iy >= g->in_h) {
                for (int kx = 0; kx < g->kw; kx++, d += NR)
                    for (int t = 0; t < run; t++) d[t] = 0.0f;
                continue;
            }
            const float *restrict row = plane + (size_t)iy * g->in_w;
            if (g->stride == 1) {
                for (int kx = 0; kx < g->kw; kx++, d += NR) {
                    const int lo = ixb + kx;         /* src ix for t = 0 */
                    int a = lo < 0 ? -lo : 0;        /* first in-image t */
                    int b = (lo + run > g->in_w) ? g->in_w - lo : run;
                    if (b < a) b = a;
                    for (int t = 0; t < a; t++)   d[t] = 0.0f;
                    for (int t = a; t < b; t++)   d[t] = row[lo + t];
                    for (int t = b; t < run; t++) d[t] = 0.0f;
                }
            } else {
                for (int kx = 0; kx < g->kw; kx++, d += NR)
                    for (int t = 0; t < run; t++) {
                        const int ix = ixb + t * g->stride + kx;
                        d[t] = (ix >= 0 && ix < g->in_w) ? row[ix] : 0.0f;
                    }
            }
        }
    }
}

/* Pack im2col columns [j0, j0+nc) into NR-column micro-panels:
 * Bp[panel][k][NR]. Column j is output pixel (sample j/npatch, patch
 * j%npatch); each micro-panel's columns are split into runs sharing
 * (sample, output row) so convg_pack_run can stream them. Columns past nc
 * are zero-filled (the kernel always runs full NR). */
static void convg_pack_B(const float *restrict X, float *restrict Bp,
                         const tk_conv_geom *g, long j0, int nc) {
    const int NR = TK_CONV_NR;
    for (int jj = 0; jj < nc; jj += NR) {
        const int jn = (nc - jj < NR) ? nc - jj : NR;
        float *restrict dst = Bp + (size_t)(jj / NR) * g->kdim * NR;
        if (jn < NR)                                  /* pad columns to NR */
            memset(dst, 0, (size_t)g->kdim * NR * sizeof(float));
        int t = 0;
        while (t < jn) {
            const long j = j0 + jj + t;
            const int s = (int)(j / g->npatch), p = (int)(j % g->npatch);
            const int oy = p / g->ow, ox = p % g->ow;
            int run = g->ow - ox;                     /* to end of output row */
            if (run > jn - t) run = jn - t;
            convg_pack_run(X + (size_t)s * g->in_sz, dst + t, run, oy, ox, g);
            t += run;
        }
    }
}

/* MR x NR micro-kernel over packed panels: Ct[MR][NR] = sum_k A[k][:] outer
 * B[k][:]. Portable C form; each accumulator is one register, fully
 * unrolled by the compiler (the local array never escapes). Fallback for
 * targets without an explicit kernel below. */
#if !defined(__aarch64__)
static void convg_kern_c(int kc, const float *restrict A,
                         const float *restrict B, float *restrict Ct) {
    float acc[TK_CONV_MR * TK_CONV_NR] = { 0 };
    for (int k = 0; k < kc; k++) {
        const float *restrict a = A + (size_t)k * TK_CONV_MR;
        const float *restrict b = B + (size_t)k * TK_CONV_NR;
        for (int i = 0; i < TK_CONV_MR; i++)
            for (int j = 0; j < TK_CONV_NR; j++)
                acc[i * TK_CONV_NR + j] += a[i] * b[j];
    }
    memcpy(Ct, acc, sizeof acc);
}
#endif /* !__aarch64__ */

#if defined(__aarch64__) && defined(TK_CONV_TILE_8X12)
/* NEON 8x12 outer-product kernel. Per k step: 2 A loads + 3 B loads feed 24
 * independent FMAs (one per accumulator), so unlike the dot-product kernels
 * there is no per-accumulator latency chain to split -- each accumulator
 * sees one FMA every ~6 cycles against ~4-cycle FMA latency (the same
 * latency-vs-throughput argument as ops.c's dual chains, satisfied here by
 * tile width instead). vfmaq_laneq broadcasts A lanes for free. */
static void convg_kern_neon(int kc, const float *restrict A,
                            const float *restrict B, float *restrict Ct) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00;
    float32x4_t c10 = c00, c11 = c00, c12 = c00;
    float32x4_t c20 = c00, c21 = c00, c22 = c00;
    float32x4_t c30 = c00, c31 = c00, c32 = c00;
    float32x4_t c40 = c00, c41 = c00, c42 = c00;
    float32x4_t c50 = c00, c51 = c00, c52 = c00;
    float32x4_t c60 = c00, c61 = c00, c62 = c00;
    float32x4_t c70 = c00, c71 = c00, c72 = c00;
    /* k unrolled x2: halves the loop overhead per 24-FMA block and lets the
     * scheduler overlap the next step's loads with this step's FMAs. Each
     * accumulator still sees strictly sequential k order (same numerics). */
#define TK__CONV_STEP(o)                                                      \
    do {                                                                      \
        const float32x4_t b0 = vld1q_f32(B + 12 * (o)),                       \
                          b1 = vld1q_f32(B + 12 * (o) + 4),                   \
                          b2 = vld1q_f32(B + 12 * (o) + 8);                   \
        const float32x4_t a0 = vld1q_f32(A + 8 * (o)),                        \
                          a1 = vld1q_f32(A + 8 * (o) + 4);                    \
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);                                \
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);                                \
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);                                \
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);                                \
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);                                \
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);                                \
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);                                \
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);                                \
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);                                \
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);                                \
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);                                \
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);                                \
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);                                \
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);                                \
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);                                \
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);                                \
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);                                \
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);                                \
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);                                \
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);                                \
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);                                \
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);                                \
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);                                \
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);                                \
    } while (0)
    int k = 0;
    for (; k + 2 <= kc; k += 2) {
        TK__CONV_STEP(0);
        TK__CONV_STEP(1);
        A += 16; B += 24;
    }
    if (k < kc) TK__CONV_STEP(0);
#undef TK__CONV_STEP
    vst1q_f32(Ct,      c00); vst1q_f32(Ct +  4, c01); vst1q_f32(Ct +  8, c02);
    vst1q_f32(Ct + 12, c10); vst1q_f32(Ct + 16, c11); vst1q_f32(Ct + 20, c12);
    vst1q_f32(Ct + 24, c20); vst1q_f32(Ct + 28, c21); vst1q_f32(Ct + 32, c22);
    vst1q_f32(Ct + 36, c30); vst1q_f32(Ct + 40, c31); vst1q_f32(Ct + 44, c32);
    vst1q_f32(Ct + 48, c40); vst1q_f32(Ct + 52, c41); vst1q_f32(Ct + 56, c42);
    vst1q_f32(Ct + 60, c50); vst1q_f32(Ct + 64, c51); vst1q_f32(Ct + 68, c52);
    vst1q_f32(Ct + 72, c60); vst1q_f32(Ct + 76, c61); vst1q_f32(Ct + 80, c62);
    vst1q_f32(Ct + 84, c70); vst1q_f32(Ct + 88, c71); vst1q_f32(Ct + 92, c72);
}

#elif defined(__aarch64__)
/* NEON 6x16 kernel (the default tile): 24 accumulators, 4 B loads + 1.5 A
 * loads per k step feed 24 independent FMAs -- one FMA per accumulator per
 * k step against ~4-cycle FMA latency, covered by tile width (the ops.c
 * dual-chain argument, satisfied structurally). */
static void convg_kern_neon(int kc, const float *restrict A,
                            const float *restrict B, float *restrict Ct) {
    float32x4_t c0[4], c1[4], c2[4], c3[4], c4[4], c5[4];
    for (int j = 0; j < 4; j++)
        c0[j] = c1[j] = c2[j] = c3[j] = c4[j] = c5[j] = vdupq_n_f32(0);
    for (int k = 0; k < kc; k++) {
        const float32x4_t a0 = vld1q_f32(A);       /* rows 0-3 */
        const float32x2_t a1 = vld1_f32(A + 4);    /* rows 4-5 */
        for (int j = 0; j < 4; j++) {
            const float32x4_t b = vld1q_f32(B + 4 * j);
            c0[j] = vfmaq_laneq_f32(c0[j], b, a0, 0);
            c1[j] = vfmaq_laneq_f32(c1[j], b, a0, 1);
            c2[j] = vfmaq_laneq_f32(c2[j], b, a0, 2);
            c3[j] = vfmaq_laneq_f32(c3[j], b, a0, 3);
            c4[j] = vfmaq_lane_f32(c4[j], b, a1, 0);
            c5[j] = vfmaq_lane_f32(c5[j], b, a1, 1);
        }
        A += 6; B += 16;
    }
    for (int j = 0; j < 4; j++) {
        vst1q_f32(Ct +      4 * j, c0[j]); vst1q_f32(Ct + 16 + 4 * j, c1[j]);
        vst1q_f32(Ct + 32 + 4 * j, c2[j]); vst1q_f32(Ct + 48 + 4 * j, c3[j]);
        vst1q_f32(Ct + 64 + 4 * j, c4[j]); vst1q_f32(Ct + 80 + 4 * j, c5[j]);
    }
}

#elif defined(__x86_64__)
#include <immintrin.h>
#define TK_CONV_HAVE_AVX2 1

/* Same cached runtime check as ops.c (private there; conv must not touch
 * ops.c): compile the AVX2 kernel unconditionally via the target attribute,
 * call it only when the CPU reports AVX2+FMA. */
static int tk__conv_cpu_avx2(void) {
    static int cached = -1;
    if (cached < 0) cached = (__builtin_cpu_supports("avx2")
                              && __builtin_cpu_supports("fma"));
    return cached;
}

/* AVX2 6x16 outer-product kernel: 12 YMM accumulators, 2 B loads + 6 A
 * broadcasts per k step feed 12 independent FMAs across ports 0/1 -- tile
 * width covers the FMA latency-throughput product exactly as the NEON
 * kernel does (and as ops.c's 8-chain dot kernel argues). */
__attribute__((target("avx2,fma")))
static void convg_kern_avx2(int kc, const float *restrict A,
                            const float *restrict B, float *restrict Ct) {
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00,
           c20 = c00, c21 = c00, c30 = c00, c31 = c00,
           c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    for (int k = 0; k < kc; k++) {
        const __m256 b0 = _mm256_loadu_ps(B), b1 = _mm256_loadu_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(A + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4);
        c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5);
        c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        A += 6; B += 16;
    }
    _mm256_storeu_ps(Ct,      c00); _mm256_storeu_ps(Ct +  8, c01);
    _mm256_storeu_ps(Ct + 16, c10); _mm256_storeu_ps(Ct + 24, c11);
    _mm256_storeu_ps(Ct + 32, c20); _mm256_storeu_ps(Ct + 40, c21);
    _mm256_storeu_ps(Ct + 48, c30); _mm256_storeu_ps(Ct + 56, c31);
    _mm256_storeu_ps(Ct + 64, c40); _mm256_storeu_ps(Ct + 72, c41);
    _mm256_storeu_ps(Ct + 80, c50); _mm256_storeu_ps(Ct + 88, c51);
}
#endif /* arch kernels */

static inline void convg_kern(int kc, const float *restrict A,
                              const float *restrict B, float *restrict Ct) {
#if defined(__aarch64__)
    convg_kern_neon(kc, A, B, Ct);        /* NEON is arm64 baseline */
#else
  #if defined(TK_CONV_HAVE_AVX2)
    if (tk__conv_cpu_avx2()) { convg_kern_avx2(kc, A, B, Ct); return; }
  #endif
    convg_kern_c(kc, A, B, Ct);
#endif
}

/* Store an m x nf tile of C into Z/Y with bias + activation. Y rows are
 * contiguous per sample (stride npatch between filter rows); a tile whose
 * columns cross a sample boundary is split at the boundary. The Z test and
 * the activation dispatch are hoisted OUT of the element loops: a
 * per-element switch (or NULL test) de-vectorizes the store, which measured
 * ~6 ms of a 16 ms forward -- the same lesson as tk_activate's switch
 * benchmark, applied to a two-destination loop. relu/identity get branchless
 * vectorizable bodies; other activations take the scalar switch. */
static void convg_store(const float *restrict Ct, float *Z, float *Y,
                        const float *bias, const tk_conv_geom *g,
                        tk_activation_t act, int oc0, int m,
                        long j0, int nf) {
    int done = 0;
    while (done < nf) {
        const long j = j0 + done;
        const int s = (int)(j / g->npatch), p = (int)(j % g->npatch);
        int chunk = g->npatch - p;
        if (chunk > nf - done) chunk = nf - done;
        const size_t base = (size_t)s * g->out_sz + p;
        for (int i = 0; i < m; i++) {
            const float b = bias ? bias[oc0 + i] : 0.0f;
            const float *restrict src = Ct + (size_t)i * TK_CONV_NR + done;
            float *restrict yr = Y + base + (size_t)(oc0 + i) * g->npatch;
            if (Z) {
                float *restrict zr = Z + base + (size_t)(oc0 + i) * g->npatch;
                switch (act) {
                case TK_ACT_RELU:
                    for (int t = 0; t < chunk; t++) {
                        const float z = src[t] + b;
                        zr[t] = z; yr[t] = __builtin_fmaxf(z, 0.0f);
                    }
                    break;
                case TK_ACT_IDENTITY:
                    for (int t = 0; t < chunk; t++) {
                        const float z = src[t] + b;
                        zr[t] = z; yr[t] = z;
                    }
                    break;
                default:
                    for (int t = 0; t < chunk; t++) {
                        const float z = src[t] + b;
                        zr[t] = z; yr[t] = tk_act_scalar(z, act);
                    }
                }
            } else {
                switch (act) {
                case TK_ACT_RELU:
                    for (int t = 0; t < chunk; t++)
                        yr[t] = __builtin_fmaxf(src[t] + b, 0.0f);
                    break;
                case TK_ACT_IDENTITY:
                    for (int t = 0; t < chunk; t++) yr[t] = src[t] + b;
                    break;
                default:
                    for (int t = 0; t < chunk; t++)
                        yr[t] = tk_act_scalar(src[t] + b, act);
                }
            }
        }
        done += chunk;
    }
}

/* Per-worker packed-B slice size: micro-panels round columns up to NR. */
static size_t conv_bp_stride(int kdim) {
    return (size_t)((TK_CONV_NC + TK_CONV_NR - 1) / TK_CONV_NR)
         * TK_CONV_NR * kdim;
}

typedef struct {
    const float *X, *bias, *Ap;
    float *Z, *Y, *Bp;           /* Bp: one conv_bp_stride slice per worker */
    tk_conv_geom g;
    tk_activation_t act;
} tk_convg_fwd_ctx;

/* Column panels [p0, p1): pack B for the panel, then sweep all filter
 * micro-panels against each B micro-panel (B micro-panel stays L1-hot). */
static void convg_fwd_worker(void *pv, int p0, int p1, int worker) {
    const tk_convg_fwd_ctx *c = (const tk_convg_fwd_ctx *)pv;
    const tk_conv_geom *g = &c->g;
    const int MR = TK_CONV_MR, NR = TK_CONV_NR;
    float *Bp = c->Bp + (size_t)worker * conv_bp_stride(g->kdim);
    for (int pan = p0; pan < p1; pan++) {
        const long j0 = (long)pan * TK_CONV_NC;
        const int nc = (g->ncols - j0 < TK_CONV_NC) ? (int)(g->ncols - j0)
                                                    : TK_CONV_NC;
        convg_pack_B(c->X, Bp, g, j0, nc);
        for (int jr = 0; jr < nc; jr += NR) {
            const float *restrict Bmp = Bp + (size_t)(jr / NR) * g->kdim * NR;
            const int nf = (nc - jr < NR) ? nc - jr : NR;
            for (int oc0 = 0; oc0 < g->out_c; oc0 += MR) {
                const int m = (g->out_c - oc0 < MR) ? g->out_c - oc0 : MR;
                float Ct[TK_CONV_MR * TK_CONV_NR];
                convg_kern(g->kdim, c->Ap + (size_t)(oc0 / MR) * g->kdim * MR,
                           Bmp, Ct);
                convg_store(Ct, c->Z, c->Y, c->bias, g, c->act,
                            oc0, m, j0 + jr, nf);
            }
        }
    }
}

/* Whole-batch GEMM forward. Returns 0 if scratch allocation failed (caller
 * falls back to the per-sample path). */
static int conv_fwd_gemm(const float *X, const float *K, const float *bias,
                         float *Z, float *Y, const tk_conv_geom *g,
                         tk_activation_t act, int T) {
    const int mpanels = (g->out_c + TK_CONV_MR - 1) / TK_CONV_MR;
    const int panels = (int)((g->ncols + TK_CONV_NC - 1) / TK_CONV_NC);
    if (panels == 1) T = 1;
    /* Worker ids span the whole pool (main thread takes the LAST id), so the
     * per-worker B scratch is sized by pool width even if panels < T. */
    const int Tw = (T > 1) ? tk_num_threads() : 1;
    float *Ap = malloc((size_t)mpanels * TK_CONV_MR * g->kdim * sizeof(float));
    float *Bp = malloc((size_t)Tw * conv_bp_stride(g->kdim) * sizeof(float));
    if (!Ap || !Bp) { free(Ap); free(Bp); return 0; }
    convg_pack_A(K, Ap, g->out_c, g->kdim);

    tk_convg_fwd_ctx c = { X, bias, Ap, Z, Y, Bp, *g, act };
    if (T > 1) tk_parallel_for(panels, convg_fwd_worker, &c);
    else       convg_fwd_worker(&c, 0, panels, 0);
    free(Ap); free(Bp);
    return 1;
}

typedef struct {
    const float *X, *K, *bias;
    float *Z, *Y, *cols;         /* cols: one npatch*kdim slice per worker */
    int in_c, in_h, in_w, out_c, kh, kw, stride, pad, oh, ow;
    tk_activation_t act;
} tk_conv_fwd_ctx;

static void conv_fwd_worker(void *p, int s0, int s1, int worker) {
    const tk_conv_fwd_ctx *c = (const tk_conv_fwd_ctx *)p;
    const int npatch = c->oh * c->ow, kdim = c->in_c * c->kh * c->kw;
    const size_t in_sz = (size_t)c->in_c * c->in_h * c->in_w;
    const size_t out_sz = (size_t)c->out_c * npatch;
    float *cols = c->cols + (size_t)worker * npatch * kdim;
    for (int s = s0; s < s1; s++) {
        im2col(c->X + (size_t)s * in_sz, cols, c->in_c, c->in_h, c->in_w,
               c->kh, c->kw, c->stride, c->pad, c->oh, c->ow);
        conv_gemm(c->K, cols, c->bias,
                  c->Z ? c->Z + (size_t)s * out_sz : NULL,
                  c->Y + (size_t)s * out_sz,
                  c->out_c, npatch, kdim, c->act);
    }
}

void tk_conv2d_forward_f32(const float *X, const float *K, const float *bias,
                           float *Z, float *Y,
                           int n, int in_c, int in_h, int in_w,
                           int out_c, int kh, int kw,
                           int stride, int pad, int act) {
    const int oh = tk_conv2d_out_dim(in_h, kh, stride, pad);
    const int ow = tk_conv2d_out_dim(in_w, kw, stride, pad);
    if (n <= 0 || in_c <= 0 || out_c <= 0 || oh <= 0 || ow <= 0) return;
    const int npatch = oh * ow, kdim = in_c * kh * kw;
    const size_t work = (size_t)n * out_c * npatch * kdim;   /* MACs per call */

    if (work >= TK_CONV_GEMM_MIN_MACS) {          /* big shapes: batch GEMM */
        const tk_conv_geom g = conv_geom(n, in_c, in_h, in_w, out_c, kh, kw,
                                         stride, pad, oh, ow);
        const int Tg = (work >= TK_MT_MIN_WORK) ? tk_num_threads() : 1;
        if (conv_fwd_gemm(X, K, bias, Z, Y, &g, (tk_activation_t)act, Tg))
            return;                               /* else: alloc failed, fall through */
    }

    const int T = (work >= TK_MT_MIN_WORK && n > 1) ? tk_num_threads() : 1;
    float *cols = malloc((size_t)T * npatch * kdim * sizeof(float));
    if (!cols) {                                  /* no scratch: direct loops */
        conv_fwd_direct(X, K, bias, Z, Y, 0, n, in_c, in_h, in_w,
                        out_c, kh, kw, stride, pad, oh, ow,
                        (tk_activation_t)act);
        return;
    }
    tk_conv_fwd_ctx c = { X, K, bias, Z, Y, cols,
                          in_c, in_h, in_w, out_c, kh, kw, stride, pad,
                          oh, ow, (tk_activation_t)act };
    if (T > 1) tk_parallel_for(n, conv_fwd_worker, &c);
    else       conv_fwd_worker(&c, 0, n, 0);
    free(cols);
}

/* One sample's backward: dKp/dbp are ACCUMULATED into (worker-private or the
 * final buffers), dxs is written. cols/dz/dcol are per-worker scratch of
 * npatch*kdim, out_c*npatch and kdim floats. */
static void conv_bwd_sample(const float *restrict xs, const float *restrict K,
                            const float *restrict zs, const float *restrict dys,
                            float *restrict dKp, float *restrict dbp,
                            float *restrict dxs,
                            float *restrict cols, float *restrict dz,
                            float *restrict dcol,
                            int in_c, int in_h, int in_w, int out_c,
                            int kh, int kw, int stride, int pad,
                            int oh, int ow, tk_activation_t act) {
    const int npatch = oh * ow, kdim = in_c * kh * kw;
    im2col(xs, cols, in_c, in_h, in_w, kh, kw, stride, pad, oh, ow);

    for (int oc = 0; oc < out_c; oc++) {          /* dz = dy * act'(z), db */
        const float *restrict zr = zs + (size_t)oc * npatch;
        const float *restrict dyr = dys + (size_t)oc * npatch;
        float *restrict dzr = dz + (size_t)oc * npatch;
        float bsum = 0.0f;
        for (int p = 0; p < npatch; p++) {
            const float g = dyr[p] * tk_act_grad_scalar(zr[p], act);
            dzr[p] = g;
            bsum += g;
        }
        if (dbp) dbp[oc] += bsum;
    }

    for (int oc = 0; oc < out_c; oc++) {          /* dK += dz . im2col rows */
        const float *restrict dzr = dz + (size_t)oc * npatch;
        float *restrict dkr = dKp + (size_t)oc * kdim;
        for (int p = 0; p < npatch; p++) {
            const float g = dzr[p];
            if (g == 0.0f) continue;              /* dead relu patch */
            const float *restrict cp = cols + (size_t)p * kdim;
            for (int j = 0; j < kdim; j++) dkr[j] += g * cp[j];
        }
    }

    if (!dxs) return;
    memset(dxs, 0, (size_t)in_c * in_h * in_w * sizeof(float));
    for (int p = 0; p < npatch; p++) {            /* dX: col2im scatter */
        memset(dcol, 0, (size_t)kdim * sizeof(float));
        int any = 0;
        for (int oc = 0; oc < out_c; oc++) {      /* dcol = K^T dz[:, p] */
            const float g = dz[(size_t)oc * npatch + p];
            if (g == 0.0f) continue;
            any = 1;
            const float *restrict kr = K + (size_t)oc * kdim;
            for (int j = 0; j < kdim; j++) dcol[j] += g * kr[j];
        }
        if (!any) continue;
        const int oy = p / ow, ox = p % ow;
        const int iy0 = oy * stride - pad, ix0 = ox * stride - pad;
        const float *restrict d = dcol;
        for (int c = 0; c < in_c; c++) {
            float *restrict plane = dxs + (size_t)c * in_h * in_w;
            for (int ky = 0; ky < kh; ky++) {
                const int iy = iy0 + ky;
                for (int kx = 0; kx < kw; kx++, d++) {
                    const int ix = ix0 + kx;
                    if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
                        plane[(size_t)iy * in_w + ix] += *d;
                }
            }
        }
    }
}

/* Scratch-free backward fallback: bounds-checked direct loops, serial. */
static void conv_bwd_direct(const float *restrict X, const float *restrict K,
                            const float *restrict Z, const float *restrict dY,
                            float *restrict dK, float *restrict db,
                            float *restrict dX,
                            int n, int in_c, int in_h, int in_w,
                            int out_c, int kh, int kw, int stride, int pad,
                            int oh, int ow, tk_activation_t act) {
    const size_t in_sz = (size_t)in_c * in_h * in_w;
    const size_t out_sz = (size_t)out_c * oh * ow;
    const int kdim = in_c * kh * kw;
    memset(dK, 0, (size_t)out_c * kdim * sizeof(float));
    if (db) memset(db, 0, (size_t)out_c * sizeof(float));
    if (dX) memset(dX, 0, (size_t)n * in_sz * sizeof(float));
    for (int s = 0; s < n; s++) {
        const float *restrict xs = X + (size_t)s * in_sz;
        float *restrict dxs = dX ? dX + (size_t)s * in_sz : NULL;
        for (int oc = 0; oc < out_c; oc++) {
            const float *restrict kf = K + (size_t)oc * kdim;
            float *restrict dkf = dK + (size_t)oc * kdim;
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    const size_t o = (size_t)s * out_sz
                                   + ((size_t)oc * oh + oy) * ow + ox;
                    const float g = dY[o] * tk_act_grad_scalar(Z[o], act);
                    if (db) db[oc] += g;
                    if (g == 0.0f) continue;
                    const int iy0 = oy * stride - pad, ix0 = ox * stride - pad;
                    for (int c = 0; c < in_c; c++)
                        for (int ky = 0; ky < kh; ky++) {
                            const int iy = iy0 + ky;
                            if (iy < 0 || iy >= in_h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                const int ix = ix0 + kx;
                                if (ix < 0 || ix >= in_w) continue;
                                const size_t kidx = ((size_t)c * kh + ky) * kw + kx;
                                const size_t xidx = ((size_t)c * in_h + iy) * in_w + ix;
                                dkf[kidx] += g * xs[xidx];
                                if (dxs) dxs[xidx] += g * kf[kidx];
                            }
                        }
                }
        }
    }
}

typedef struct {
    const float *X, *K, *Z, *dY;
    float *dKp, *dbp, *dX;       /* dKp/dbp: one zeroed slice per worker */
    float *cols, *dz, *dcol;     /* per-worker scratch */
    int in_c, in_h, in_w, out_c, kh, kw, stride, pad, oh, ow;
    tk_activation_t act;
} tk_conv_bwd_ctx;

static void conv_bwd_worker(void *p, int s0, int s1, int worker) {
    const tk_conv_bwd_ctx *c = (const tk_conv_bwd_ctx *)p;
    const int npatch = c->oh * c->ow, kdim = c->in_c * c->kh * c->kw;
    const size_t in_sz = (size_t)c->in_c * c->in_h * c->in_w;
    const size_t out_sz = (size_t)c->out_c * npatch;
    float *dKp = c->dKp + (size_t)worker * c->out_c * kdim;
    float *dbp = c->dbp ? c->dbp + (size_t)worker * c->out_c : NULL;
    float *cols = c->cols + (size_t)worker * npatch * kdim;
    float *dz = c->dz + (size_t)worker * out_sz;
    float *dcol = c->dcol + (size_t)worker * kdim;
    for (int s = s0; s < s1; s++)
        conv_bwd_sample(c->X + (size_t)s * in_sz, c->K,
                        c->Z + (size_t)s * out_sz, c->dY + (size_t)s * out_sz,
                        dKp, dbp, c->dX ? c->dX + (size_t)s * in_sz : NULL,
                        cols, dz, dcol,
                        c->in_c, c->in_h, c->in_w, c->out_c,
                        c->kh, c->kw, c->stride, c->pad, c->oh, c->ow, c->act);
}

void tk_conv2d_backward_f32(const float *X, const float *K, const float *Z,
                            const float *dY,
                            float *dK, float *db, float *dX,
                            int n, int in_c, int in_h, int in_w,
                            int out_c, int kh, int kw,
                            int stride, int pad, int act) {
    const int oh = tk_conv2d_out_dim(in_h, kh, stride, pad);
    const int ow = tk_conv2d_out_dim(in_w, kw, stride, pad);
    if (in_c <= 0 || out_c <= 0 || kh <= 0 || kw <= 0) return;
    const int npatch = oh * ow, kdim = in_c * kh * kw;
    if (n <= 0 || npatch <= 0) {                  /* empty batch: zero sums */
        memset(dK, 0, (size_t)out_c * kdim * sizeof(float));
        if (db) memset(db, 0, (size_t)out_c * sizeof(float));
        if (dX && n > 0)
            memset(dX, 0, (size_t)n * in_c * in_h * in_w * sizeof(float));
        return;
    }

    const size_t work = (size_t)n * out_c * npatch * kdim;
    const int T = (work >= TK_MT_MIN_WORK && n > 1) ? tk_num_threads() : 1;
    /* Per-worker: im2col + dz + dcol scratch, and zeroed dK/db partials for
     * the batch-summed reduction (same pattern as tk_linear_backward's dx). */
    const size_t out_sz = (size_t)out_c * npatch;
    float *cols = malloc((size_t)T * ((size_t)npatch * kdim + out_sz + kdim)
                         * sizeof(float));
    float *part = calloc((size_t)T * ((size_t)out_c * kdim + (db ? out_c : 0)),
                         sizeof(float));
    if (!cols || !part) {
        free(cols); free(part);
        conv_bwd_direct(X, K, Z, dY, dK, db, dX, n, in_c, in_h, in_w,
                        out_c, kh, kw, stride, pad, oh, ow,
                        (tk_activation_t)act);
        return;
    }
    float *dz = cols + (size_t)T * npatch * kdim;
    float *dcol = dz + (size_t)T * out_sz;
    float *dbp = db ? part + (size_t)T * out_c * kdim : NULL;

    tk_conv_bwd_ctx c = { X, K, Z, dY, part, dbp, dX, cols, dz, dcol,
                          in_c, in_h, in_w, out_c, kh, kw, stride, pad,
                          oh, ow, (tk_activation_t)act };
    if (T > 1) tk_parallel_for(n, conv_bwd_worker, &c);
    else       conv_bwd_worker(&c, 0, n, 0);

    memcpy(dK, part, (size_t)out_c * kdim * sizeof(float));
    for (int t = 1; t < T; t++) {                 /* reduce worker partials */
        const float *pt = part + (size_t)t * out_c * kdim;
        for (size_t j = 0; j < (size_t)out_c * kdim; j++) dK[j] += pt[j];
    }
    if (db) {
        memcpy(db, dbp, (size_t)out_c * sizeof(float));
        for (int t = 1; t < T; t++) {
            const float *pt = dbp + (size_t)t * out_c;
            for (int j = 0; j < out_c; j++) db[j] += pt[j];
        }
    }
    free(cols); free(part);
}

void tk_maxpool2d_f32(const float *X, float *Y, int32_t *argmax,
                      int n, int c, int in_h, int in_w,
                      int pool, int stride) {
    /* Floor semantics, no padding: a ragged right/bottom edge narrower than
     * the window is dropped (documented in conv.h). */
    const int oh = tk_conv2d_out_dim(in_h, pool, stride, 0);
    const int ow = tk_conv2d_out_dim(in_w, pool, stride, 0);
    if (n <= 0 || c <= 0 || oh <= 0 || ow <= 0) return;
    const size_t planes = (size_t)n * c;
    for (size_t pl = 0; pl < planes; pl++) {
        const float *restrict xp = X + pl * (size_t)in_h * in_w;
        float *restrict yp = Y + pl * (size_t)oh * ow;
        int32_t *restrict ap = argmax + pl * (size_t)oh * ow;
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                const int iy0 = oy * stride, ix0 = ox * stride;
                int best = iy0 * in_w + ix0;
                float m = xp[best];
                for (int py = 0; py < pool; py++) {
                    const int base = (iy0 + py) * in_w + ix0;
                    for (int px = 0; px < pool; px++)
                        if (xp[base + px] > m) { m = xp[base + px]; best = base + px; }
                }
                yp[(size_t)oy * ow + ox] = m;
                ap[(size_t)oy * ow + ox] = (int32_t)best;
            }
    }
}

void tk_maxpool2d_backward_f32(const float *dY, const int32_t *argmax,
                               float *dX,
                               int n, int c, int in_h, int in_w,
                               int out_h, int out_w) {
    if (n <= 0 || c <= 0 || in_h <= 0 || in_w <= 0) return;
    const size_t planes = (size_t)n * c, opix = (size_t)out_h * out_w;
    memset(dX, 0, planes * (size_t)in_h * in_w * sizeof(float));
    if (out_h <= 0 || out_w <= 0) return;
    for (size_t pl = 0; pl < planes; pl++) {
        const float *restrict dyp = dY + pl * opix;
        const int32_t *restrict ap = argmax + pl * opix;
        float *restrict dxp = dX + pl * (size_t)in_h * in_w;
        for (size_t i = 0; i < opix; i++) dxp[ap[i]] += dyp[i];
    }
}

/* --- batched float32 dense head ------------------------------------------ */

/* Rows [o0, o1) against all samples; sample-inner keeps the 4-row W block hot
 * in L1 across the batch, the gemm_range argument from ops.c. */
static void linb_fwd_range(const float *restrict W, const float *restrict X,
                           const float *restrict bias,
                           float *restrict Z, float *restrict Y,
                           int o0, int o1, int n, int out_dim, int in_dim,
                           tk_activation_t act) {
    int o = o0;
    for (; o + 4 <= o1; o += 4) {
        for (int s = 0; s < n; s++) {
            float r[4];
            tk__dot4_f32(W + (size_t)o * in_dim, in_dim,
                         X + (size_t)s * in_dim, r);
            for (int k = 0; k < 4; k++) {
                const float z = r[k] + (bias ? bias[o + k] : 0.0f);
                if (Z) Z[(size_t)s * out_dim + o + k] = z;
                Y[(size_t)s * out_dim + o + k] = tk_act_scalar(z, act);
            }
        }
    }
    for (; o < o1; o++) {
        const float *restrict wr = W + (size_t)o * in_dim;
        const float b = bias ? bias[o] : 0.0f;
        for (int s = 0; s < n; s++) {
            const float z = b + tk__dot1_f32(wr, X + (size_t)s * in_dim, in_dim);
            if (Z) Z[(size_t)s * out_dim + o] = z;
            Y[(size_t)s * out_dim + o] = tk_act_scalar(z, act);
        }
    }
}

typedef struct {
    const float *W, *X, *bias;
    float *Z, *Y;
    int n, out_dim, in_dim;
    tk_activation_t act;
} tk_linb_ctx;

static void linb_fwd_worker(void *p, int o0, int o1, int worker) {
    (void)worker;
    const tk_linb_ctx *c = (const tk_linb_ctx *)p;
    linb_fwd_range(c->W, c->X, c->bias, c->Z, c->Y, o0, o1,
                   c->n, c->out_dim, c->in_dim, c->act);
}

void tk_linear_forward_batch_f32(const float *W, const float *X,
                                 const float *bias, float *Z, float *Y,
                                 int n, int out_dim, int in_dim, int act) {
    if (n <= 0 || out_dim <= 0) return;
    if ((size_t)n * out_dim * in_dim >= TK_MT_MIN_WORK && tk_num_threads() > 1) {
        tk_linb_ctx c = { W, X, bias, Z, Y, n, out_dim, in_dim,
                          (tk_activation_t)act };
        tk_parallel_for(out_dim, linb_fwd_worker, &c);
        return;
    }
    linb_fwd_range(W, X, bias, Z, Y, 0, out_dim, n, out_dim, in_dim,
                   (tk_activation_t)act);
}

/* dW rows [o0, o1) + db, summed over the batch (rows disjoint, no locking). */
static void linb_bwd_rows(const float *restrict X, const float *restrict Z,
                          const float *restrict dY,
                          float *restrict dW, float *restrict db,
                          int o0, int o1, int n, int out_dim, int in_dim,
                          tk_activation_t act) {
    for (int o = o0; o < o1; o++) {
        float *restrict dwr = dW + (size_t)o * in_dim;
        memset(dwr, 0, (size_t)in_dim * sizeof(float));
        float bsum = 0.0f;
        for (int s = 0; s < n; s++) {
            const float g = dY[(size_t)s * out_dim + o]
                          * tk_act_grad_scalar(Z[(size_t)s * out_dim + o], act);
            bsum += g;
            if (g == 0.0f) continue;
            const float *restrict xs = X + (size_t)s * in_dim;
            for (int i = 0; i < in_dim; i++) dwr[i] += g * xs[i];
        }
        if (db) db[o] = bsum;
    }
}

/* dX rows for samples [s0, s1) (samples disjoint; dz recomputed, cheap). */
static void linb_bwd_dx(const float *restrict W, const float *restrict Z,
                        const float *restrict dY, float *restrict dX,
                        int s0, int s1, int out_dim, int in_dim,
                        tk_activation_t act) {
    for (int s = s0; s < s1; s++) {
        float *restrict dxs = dX + (size_t)s * in_dim;
        memset(dxs, 0, (size_t)in_dim * sizeof(float));
        for (int o = 0; o < out_dim; o++) {
            const float g = dY[(size_t)s * out_dim + o]
                          * tk_act_grad_scalar(Z[(size_t)s * out_dim + o], act);
            if (g == 0.0f) continue;
            const float *restrict wr = W + (size_t)o * in_dim;
            for (int i = 0; i < in_dim; i++) dxs[i] += g * wr[i];
        }
    }
}

typedef struct {
    const float *W, *X, *Z, *dY;
    float *dW, *db, *dX;
    int n, out_dim, in_dim;
    tk_activation_t act;
} tk_linb_bwd_ctx;

static void linb_bwd_rows_worker(void *p, int o0, int o1, int worker) {
    (void)worker;
    const tk_linb_bwd_ctx *c = (const tk_linb_bwd_ctx *)p;
    linb_bwd_rows(c->X, c->Z, c->dY, c->dW, c->db, o0, o1,
                  c->n, c->out_dim, c->in_dim, c->act);
}

static void linb_bwd_dx_worker(void *p, int s0, int s1, int worker) {
    (void)worker;
    const tk_linb_bwd_ctx *c = (const tk_linb_bwd_ctx *)p;
    linb_bwd_dx(c->W, c->Z, c->dY, c->dX, s0, s1,
                c->out_dim, c->in_dim, c->act);
}

void tk_linear_backward_batch_f32(const float *W, const float *X,
                                  const float *Z, const float *dY,
                                  float *dW, float *db, float *dX,
                                  int n, int out_dim, int in_dim, int act) {
    if (out_dim <= 0 || in_dim <= 0) return;
    if (n <= 0) {                                 /* empty batch: zero sums */
        memset(dW, 0, (size_t)out_dim * in_dim * sizeof(float));
        if (db) memset(db, 0, (size_t)out_dim * sizeof(float));
        return;
    }
    tk_linb_bwd_ctx c = { W, X, Z, dY, dW, db, dX, n, out_dim, in_dim,
                          (tk_activation_t)act };
    const int mt = (size_t)n * out_dim * in_dim >= TK_MT_MIN_WORK
                 && tk_num_threads() > 1;
    if (mt) tk_parallel_for(out_dim, linb_bwd_rows_worker, &c);
    else    linb_bwd_rows(X, Z, dY, dW, db, 0, out_dim, n, out_dim, in_dim,
                          (tk_activation_t)act);
    if (dX) {
        if (mt) tk_parallel_for(n, linb_bwd_dx_worker, &c);
        else    linb_bwd_dx(W, Z, dY, dX, 0, n, out_dim, in_dim,
                            (tk_activation_t)act);
    }
}

float tk_softmax_xent_f32(const float *logits, const int32_t *labels,
                          float *dlogits, int n, int classes) {
    if (n <= 0 || classes <= 0) return 0.0f;
    const float invn = 1.0f / (float)n;
    double sum = 0.0;                             /* mean over n: f64 keeps it exact */
    for (int s = 0; s < n; s++) {
        const float *restrict row = logits + (size_t)s * classes;
        float *restrict drow = dlogits + (size_t)s * classes;
        float m = row[0];
        for (int j = 1; j < classes; j++) if (row[j] > m) m = row[j];
        float den = 0.0f;
        for (int j = 0; j < classes; j++) {       /* stash exp(z - max) in drow */
            drow[j] = expf(row[j] - m);
            den += drow[j];
        }
        const int y = (int)labels[s];
        const float inv_den = 1.0f / den;
        for (int j = 0; j < classes; j++)
            drow[j] = (drow[j] * inv_den - (j == y ? 1.0f : 0.0f)) * invn;
        sum += (double)(logf(den) - (row[y] - m));
    }
    return (float)(sum / n);
}

void tk_sgd_update_f32(float *W, const float *dW, int n, float lr) {
    for (int i = 0; i < n; i++) W[i] -= lr * dW[i];
}
