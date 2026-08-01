/* Vectorised narrow reads must agree with the scalar reads bit-for-bit.
 *
 * The NEON loaders are static inline inside ops.c because they are only ever
 * called from the dot kernel, so this test includes that translation unit
 * rather than a copy of the arithmetic -- a copy could drift from what the
 * library actually compiles. Build it once per storage type; the body compiles
 * to a skip for types that have no vectorised loader. */
#include "../src/ops.c"

#include <stdio.h>

#if defined(TK_HAVE_NEON_FP4)
/* Only sixteen distinct inputs, but the stored byte carries the value in its low
 * nibble, so sweep all 256 byte values to pin the masking too. */
static int check_fp4(void) {
    int bad = 0;
    for (int lane = 0; lane < 4; lane++) {
        for (int v = 0; v < 256; v++) {
            tk_fp4_t buf[4] = {0, 0, 0, 0};
            buf[lane] = (tk_fp4_t)v;
            float got[4];
            vst1q_f32(got, tk__fp4x4(buf));
            for (int k = 0; k < 4; k++) {
                float want = tk_fp4_to_float(buf[k]);
                uint32_t g, w;
                memcpy(&g, &got[k], 4);
                memcpy(&w, &want, 4);
                if (g != w) {
                    if (bad < 5)
                        printf("  FAIL v=%3d lane=%d k=%d: neon=%08x scalar=%08x\n", v, lane, k, g, w);
                    bad++;
                }
            }
        }
    }
    printf("  fp4 NEON read: 1024 patterns checked, %d mismatches\n", bad);
    return bad;
}
#endif

#if defined(TK_HAVE_NEON_F8)
/* Every pattern in every lane: a loader that mishandled one lane, or that
 * worked only for the low byte of the 32-bit load, would pass a single-lane
 * check. */
static int check_f8(void) {
    int bad = 0;
    for (int lane = 0; lane < 4; lane++) {
        for (int v = 0; v < 256; v++) {
            tk_f8_t buf[4] = {0, 0, 0, 0};
            buf[lane] = (tk_f8_t)v;
            float got[4];
            vst1q_f32(got, tk__f8x4(buf));
            for (int k = 0; k < 4; k++) {
                float want = tk_f8_to_float(buf[k]);
                uint32_t g, w;
                memcpy(&g, &got[k], 4);
                memcpy(&w, &want, 4);
                if (g != w) {
                    if (bad < 5)
                        printf("  FAIL v=%3d lane=%d k=%d: neon=%08x scalar=%08x\n", v, lane, k, g, w);
                    bad++;
                }
            }
        }
    }
    printf("  tekin8 NEON read: 1024 patterns checked, %d mismatches\n", bad);
    return bad;
}
#endif


/* Block scaling must never make the dot product's scale factor underflow.
 * tk__dot_blocked multiplies a block's partial sum by scale_w * scale_x, so if
 * a format's scale sits near the 2^-126 clamp that product is zero in float32
 * and every output silently becomes zero -- which is what happened to bfloat16
 * before wide formats stopped being scaled at all. */
static int check_block_scale(void) {
    int bad = 0;
    const float amaxes[] = {1e-8f, 1e-4f, 0.05f, 0.46f, 1.0f, 7.5f, 1e3f, 1e8f};
    for (unsigned t = 0; t < sizeof amaxes / sizeof *amaxes; t++) {
        uint8_t b = tk_e8m0_from_amax(amaxes[t]);
        float sc = tk_e8m0_to_float(b);
        if (!(sc > 0.0f) || sc * sc == 0.0f) {
            printf("  FAIL amax=%g: scale=%g, scale^2 underflows\n", (double)amaxes[t], (double)sc);
            bad++;
            continue;
        }
        if (TK_BLOCK_SCALED && amaxes[t] / sc > TK_MAX_MAG) {
            printf("  FAIL amax=%g: scaled to %g, over TK_MAX_MAG %g\n",
                   (double)amaxes[t], (double)(amaxes[t] / sc), (double)TK_MAX_MAG);
            bad++;
        }
        if (!TK_BLOCK_SCALED && sc != 1.0f) {
            printf("  FAIL wide format should not scale, got %g\n", (double)sc);
            bad++;
        }
    }
    printf("  block scale (%s): %u magnitudes checked, %d bad\n",
           TK_BLOCK_SCALED ? "active" : "no-op for this format",
           (unsigned)(sizeof amaxes / sizeof *amaxes), bad);
    return bad;
}

int main(void) {
    int bad = 0;
    printf("SIMD read equivalence (%s)\n", tk_dtype_name());
    bad += check_block_scale();
#if defined(TK_HAVE_NEON_F8)
    bad += check_f8();
#elif defined(TK_HAVE_NEON_FP4)
    bad += check_fp4();
#else
    printf("  no vectorised loader for this storage type -- nothing to check\n");
#endif
    if (bad) {
        printf("FAILED: %d mismatch(es)\n", bad);
        return 1;
    }
    printf("OK\n");
    return 0;
}
