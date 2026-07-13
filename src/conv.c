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
