#include "dtypes.h"
#include <math.h>

/* float32 -> narrow. These run once when weights are quantized/loaded, so they
 * favor clarity and correctness over raw speed. The reverse (hot) direction is
 * the branchless inline code in dtypes.h. A production hot path would use
 * hardware conversions (F16C _mm256_cvtps_ph for fp16, AVX512-BF16 for bf16). */

tk_bf16_t tk_float_to_bf16(float f) {
    uint32_t x = tk__f2u(f);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return (tk_bf16_t)((x >> 16) | 0x0040u); /* keep NaN */
    x += 0x7FFFu + ((x >> 16) & 1u);          /* round to nearest, ties to even */
    return (tk_bf16_t)(x >> 16);
}

tk_fp16_t tk_float_to_fp16(float f) {
    const uint32_t sign = signbit(f) ? 0x8000u : 0u;
    const float af = fabsf(f);
    if (isnan(af))                return (tk_fp16_t)(sign | 0x7E00u);
    if (isinf(af) || af >= 65520.0f) return (tk_fp16_t)(sign | 0x7C00u); /* 65520 rounds to inf */
    if (af == 0.0f)               return (tk_fp16_t)sign;

    int e; float m = frexpf(af, &e); m *= 2.0f; e -= 1;   /* af = m * 2^e, m in [1,2) */
    int ef = e + 15;
    if (ef >= 1) {                                        /* normal */
        long mant = lroundf((m - 1.0f) * 1024.0f);
        if (mant == 1024) { mant = 0; ef++; if (ef >= 31) return (tk_fp16_t)(sign | 0x7C00u); }
        return (tk_fp16_t)(sign | ((uint32_t)ef << 10) | ((uint32_t)mant & 0x3FFu));
    }
    long mant = lroundf(af * 16777216.0f);               /* subnormal: round(af * 2^24) */
    if (mant <= 0)   return (tk_fp16_t)sign;
    if (mant > 1023) return (tk_fp16_t)(sign | (1u << 10));
    return (tk_fp16_t)(sign | ((uint32_t)mant & 0x3FFu));
}

tk_t32_t tk_float_to_t32(float f) {
    const uint32_t x = tk__f2u(f);
    const uint32_t s = x & 0x80000000u;
    const uint32_t e = (x >> 23) & 0xFFu;
    const uint32_t m = x & 0x7FFFFFu;
    if (e == 0xFFu) return s | (0x7Fu << 24) | (m ? 0x800000u : 0u);  /* inf / nan */

    int te = (int)e - 127 + 63;                          /* rebias 127 -> 63 */
    if (te >= 127) return s | (0x7Fu << 24);             /* overflow -> inf */
    if (te >= 1)   return s | ((uint32_t)te << 24) | (m << 1); /* 23 -> 24: lossless pad */

    /* subnormal: value = |f| * 2^86, held in 24-bit mantissa */
    long r = lroundf(fabsf(f) * 7.737125245533626e25f);  /* 2^86 */
    if (r <= 0)         return s;
    if (r >= (1 << 24)) return s | (1u << 24);           /* rounded up to smallest normal */
    return s | (uint32_t)r;
}

tk_f8_t tk_float_to_f8(float f) {
    const uint32_t sbit = (signbit(f) ? 1u : 0u) << 7;
    const float af = fabsf(f);
    if (af == 0.0f)               return (tk_f8_t)sbit;
    if (isnan(af) || isinf(af))   return (tk_f8_t)(sbit | (0xFu << 3) | 0x7u); /* no inf: clamp */

    int e; float m = frexpf(af, &e); m *= 2.0f; e -= 1;
    int ef = e + 7;
    if (ef >= 15) return (tk_f8_t)(sbit | (0xFu << 3) | 0x7u);        /* clamp to max (=480) */
    if (ef <= 0) {                                                    /* subnormal: (mant/8)*2^-6 */
        long r = lroundf(af * 512.0f);                               /* af * 2^6 * 8 */
        if (r <= 0) return (tk_f8_t)sbit;
        if (r > 7)  return (tk_f8_t)(sbit | (1u << 3));
        return (tk_f8_t)(sbit | (uint32_t)r);
    }
    long mant = lroundf((m - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; ef++; if (ef >= 15) return (tk_f8_t)(sbit | (0xFu << 3) | 0x7u); }
    return (tk_f8_t)(sbit | ((uint32_t)ef << 3) | ((uint32_t)mant & 0x7u));
}

tk_e5m2_t tk_float_to_e5m2(float f) {
    const uint32_t sbit = (signbit(f) ? 1u : 0u) << 7;
    const float af = fabsf(f);
    if (isnan(af))              return (tk_e5m2_t)(sbit | 0x7Eu);        /* nan */
    if (isinf(af) || af >= 61440.0f) return (tk_e5m2_t)(sbit | 0x7Cu);  /* inf (max normal 57344) */
    if (af == 0.0f)             return (tk_e5m2_t)sbit;

    int e; float m = frexpf(af, &e); m *= 2.0f; e -= 1;
    int ef = e + 15;
    if (ef >= 1) {
        long mant = lroundf((m - 1.0f) * 4.0f);
        if (mant == 4) { mant = 0; ef++; if (ef >= 31) return (tk_e5m2_t)(sbit | 0x7Cu); }
        return (tk_e5m2_t)(sbit | ((uint32_t)ef << 2) | ((uint32_t)mant & 3u));
    }
    long mant = lroundf(af * 65536.0f);                                 /* subnormal: round(af * 2^16) */
    if (mant <= 0) return (tk_e5m2_t)sbit;
    if (mant > 3)  return (tk_e5m2_t)(sbit | (1u << 2));
    return (tk_e5m2_t)(sbit | ((uint32_t)mant & 3u));
}

tk_fp4_t tk_float_to_fp4(float f) {
    /* Only 8 magnitudes exist; nearest-value search is exact and clearest. */
    static const float lv[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const uint32_t sbit = (signbit(f) ? 1u : 0u) << 3;
    const float af = fabsf(f);
    if (isnan(af) || isinf(af)) return (tk_fp4_t)(sbit | 7u);           /* no inf/nan: clamp to 6 */
    int best = 0; float bd = af;                                        /* |af - 0| */
    for (int i = 1; i < 8; i++) {
        float d = fabsf(af - lv[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return (tk_fp4_t)(sbit | (uint32_t)best);
}

const char *tk_dtype_name(void) { return TK_DTYPE_NAME; }
int         tk_scalar_size(void) { return (int)sizeof(tk_scalar_t); }
