/* Vectorised narrow reads must agree with the scalar reads bit-for-bit.
 *
 * The NEON loaders are static inline inside ops.c because they are only ever
 * called from the dot kernel, so this test includes that translation unit
 * rather than a copy of the arithmetic -- a copy could drift from what the
 * library actually compiles. Build it once per storage type; the body compiles
 * to a skip for types that have no vectorised loader. */
#include "../src/ops.c"

#include <stdio.h>

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

int main(void) {
    int bad = 0;
    printf("SIMD read equivalence (%s)\n", tk_dtype_name());
#if defined(TK_HAVE_NEON_F8)
    bad += check_f8();
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
