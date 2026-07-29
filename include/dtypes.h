#ifndef MANTISSA_DTYPES_H
#define MANTISSA_DTYPES_H

#include <stdint.h>
#include <string.h>
#include "config.h"
#include "tk_export.h"

#if !defined(__aarch64__) && defined(__F16C__)
#include <immintrin.h>   /* _mm_cvtph_ps for the fp16 read path */
#endif

/* Narrow storage formats held in plain integers. See docs/DESIGN.md for the
 * bit layouts and the rationale behind tekin32 / tekin8.
 *
 *   fp16     1-5-10  IEEE-754 half   (IEEE 754-2008 binary16)
 *   bf16     1-8-7   Google bfloat16 (Kalamkar et al., 2019, arXiv:1905.12322)
 *   tekin32  1-7-24  range traded for precision, tuned to NN value ranges
 *   tekin8   1-4-3   FP8 E4M3        (Micikevicius et al., 2022, arXiv:2209.05433)
 *   e5m2     1-5-2   FP8 E5M2, the range-favoring FP8 (same paper; OCP/IEEE P3109)
 *   fp4      1-2-1   FP4 E2M1, the MXFP4/NVFP4 element type (OCP MX v1.0, 2023)
 */
typedef uint16_t tk_fp16_t;
typedef uint16_t tk_bf16_t;
typedef uint32_t tk_t32_t;
typedef uint8_t  tk_f8_t;
typedef uint8_t  tk_e5m2_t;
typedef uint8_t  tk_fp4_t;   /* value lives in the low 4 bits */

/* type-pun without violating strict aliasing (compiles to a no-op at -O2). */
static inline uint32_t tk__f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float    tk__u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- READ path: narrow -> float32 (hot loop, branchless where it counts) --- */

static inline float tk_bf16_to_float(tk_bf16_t v) {
    return tk__u2f((uint32_t)v << 16);            /* bf16 is just the top 16 bits */
}

static inline float tk_fp16_to_float(tk_fp16_t h) {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    /* Single FCVT (ARMv8 base ISA): branchless, IEEE-exact incl. subnormals.
     * The software path below is ~10 insts + 2 branches per element, which
     * throttles the fp16 GEMV far below memory bandwidth. */
    __fp16 x; memcpy(&x, &h, 2); return (float)x;
#elif defined(__F16C__)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));  /* vcvtph2ps */
#else
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x3FFu;
    if (e == 0u)                                   /* zero or subnormal (m*2^-24) */
        return tk__u2f(s | tk__f2u((float)m * (1.0f / 16777216.0f)));
    if (e == 0x1Fu)                                /* inf / nan */
        return tk__u2f(s | 0x7F800000u | (m << 13));
    return tk__u2f(s | ((e + 112u) << 23) | (m << 13));   /* bias 15 -> 127 */
#endif
}

static inline float tk_t32_to_float(tk_t32_t v) {
    const uint32_t s = v & 0x80000000u;
    const uint32_t e = (v >> 24) & 0x7Fu;
    uint32_t m = v & 0xFFFFFFu;                    /* 24 mantissa bits */
    if (e == 0u) {                                 /* zero or subnormal (m*2^-86) */
        if (m == 0u) return tk__u2f(s);
        return tk__u2f(s | tk__f2u((float)m * 1.2924697071141057e-26f)); /* *2^-86 */
    }
    if (e == 0x7Fu) return tk__u2f(s | 0x7F800000u | (m ? 0x400000u : 0u));
    uint32_t fm = m >> 1;                          /* 24 -> 23: drop 1 bit, RNE */
    fm += (m & 1u) & (fm & 1u);
    uint32_t fe = e + 64u;                         /* bias 63 -> 127 */
    if (fm == 0x800000u) { fm = 0u; fe++; }
    return tk__u2f(s | (fe << 23) | fm);
}

/* Branchless: place the 7-bit (exp:mantissa) field as if it were already a
 * float32 significand (mag << (23-3)), then multiply by the constant
 * 2^(254-bias) = 2^120. Multiplying a float by a power of two is exact and
 * shifts the *value* of both the normal and the subnormal encoding alike --
 * the same trick as the classic branch-free fp16->fp32 widen (Giesen,
 * "Fast Half Float Conversions", 2012), generalized from bias 15 to bias 7.
 * It only stays a pure multiply because tekin8 reserves no S.1111.111 NaN
 * slot (docs/DESIGN.md): every one of the 256 patterns is a finite value, so
 * there is no inf/nan case to special-case. Verified bit-identical to the
 * branchy form over all 256 patterns. This is the fix for "the tekin8
 * lesson" (docs/PERFORMANCE.md): the subnormal branch was exactly what kept
 * the narrow read off the vectorizer. */
static inline float tk_f8_to_float(tk_f8_t v) {
    const uint32_t bits = ((uint32_t)(v & 0x80u) << 24) | ((uint32_t)(v & 0x7Fu) << 20);
    return tk__u2f(bits) * tk__u2f(247u << 23);   /* 2^(254-7) */
}

static inline float tk_e5m2_to_float(tk_e5m2_t v) {
    const uint32_t s = (v >> 7) & 1u, e = (v >> 2) & 0x1Fu, m = v & 3u;
    if (e == 0u) {                                 /* subnormal: m * 2^-16 */
        const float sub = (float)m * (1.0f / 65536.0f);
        return s ? -sub : sub;
    }
    if (e == 0x1Fu)                                /* inf / nan (shares fp16's 5-bit exp) */
        return tk__u2f((s << 31) | 0x7F800000u | (m << 21));
    return tk__u2f((s << 31) | ((e + 112u) << 23) | (m << 21));  /* bias 15 -> 127 */
}

/* E2M1 has no inf/nan and only 16 encodings, so its entire value set is a
 * 64-byte table that stays in L1 for the life of the process: the read is one
 * load, replacing a shift/mask, a select on the subnormal case, and a sign
 * fix-up. Entries are the same expression the arithmetic form computed --
 * magnitudes 0,.5,1,1.5,2,3,4,6 indexed by (e<<1)|m, negated for s -- including
 * the -0.0 at index 8, so the conversion is bit-identical for all 16 patterns.
 * Measured 2.0x on the fp4 GEMV, flat from 512 to 8192 (0.25 MB to 64 MB): the
 * bound was the conversion, not the memory traffic. */
static inline float tk_fp4_to_float(tk_fp4_t v) {
    static const float lut[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[v & 15u];
}

/* ---- WRITE path: float32 -> narrow ----------------------------------------
 *
 * "Runs once at weight load" holds for the narrow API but not for the float32
 * one: ops.c's tk_q round-trips EVERY weight through the storage grid on every
 * tk_linear_forward_f32 pass, and tk_quantize is a bulk conversion. For the two
 * formats whose converter was libm-shaped (frexpf/lroundf), that call boundary
 * blocked vectorisation outright -- the loop ran one element per iteration
 * around a `bl`. Those two get a static inline bit-arithmetic writer here and
 * dtypes.c's exported symbol forwards to it, so the inlined and exported forms
 * are the same code.
 *
 * Both are bit-identical to the libm originals over all 2^32 float patterns
 * (verified exhaustively against the pre-change functions, comparing returned
 * patterns so signed zero and NaN payloads count). Two facts make that
 * reproducible rather than lucky: frexpf's m-1 and the following power-of-two
 * scale are exact, so lroundf sees an exact value and its round-half-AWAY-from-
 * zero becomes "add half an ulp, then truncate" in integer form; and a mantissa
 * carry out of that add lands exactly on the next binade's first grid point, so
 * the reference's manual exponent bump falls out of the same add. Note this is
 * NOT the hardware FCVT rounding, which is round-half-to-EVEN and differs from
 * the reference on 0.39% of inputs -- see docs/DESIGN.md.
 *
 * bf16 is deliberately left on its exported function: its converter was already
 * branchless bit arithmetic, and inlining it measured no change (the float32
 * ladder came out at 1.007x, inside a 1.049x A/A noise band), so there is
 * nothing to buy for the extra code. t32/e5m2/fp4 keep the libm forms too --
 * each would need its own exhaustive proof and no benchmark exercises them hot. */

static inline tk_fp16_t tk__fp16_from_float(float f) {
    const uint32_t x = tk__f2u(f);
    const uint32_t s = (x >> 16) & 0x8000u;
    const uint32_t a = x & 0x7FFFFFFFu;                  /* |f| */
    if (a > 0x7F800000u)  return (tk_fp16_t)(s | 0x7E00u);   /* nan */
    if (a >= 0x477FF000u) return (tk_fp16_t)(s | 0x7C00u);   /* >= 65520, inf */
    if (a >= 0x38800000u)                                    /* |f| >= 2^-14 */
        return (tk_fp16_t)(s | (((a + 0x1000u) >> 13) - 0x1C000u));
    const uint32_t sh = 126u - (a >> 23);                /* subnormal: |f|*2^24 */
    if ((a >> 23) == 0u || sh > 25u) return (tk_fp16_t)s; /* rounds to +-0 */
    return (tk_fp16_t)(s | (((0x800000u | (a & 0x7FFFFFu)) + (1u << (sh - 1u))) >> sh));
}

static inline tk_f8_t tk__f8_from_float(float f) {
    const uint32_t x = tk__f2u(f);
    const uint32_t s = (x >> 24) & 0x80u;
    const uint32_t a = x & 0x7FFFFFFFu;
    if (a >= 0x43800000u) return (tk_f8_t)(s | 0x7Fu);   /* >= 2^8, inf, nan */
    if (a >= 0x3C800000u) {                              /* |f| >= 2^-6 */
        const uint32_t r = (a + 0x80000u) >> 20;         /* round, carry included */
        return (tk_f8_t)(r >= (135u << 3) ? (s | 0x7Fu) : (s | (r - 960u)));
    }
    const uint32_t sh = 141u - (a >> 23);                /* subnormal: |f|*2^9 */
    if ((a >> 23) == 0u || sh > 25u) return (tk_f8_t)s;
    return (tk_f8_t)(s | (((0x800000u | (a & 0x7FFFFFu)) + (1u << (sh - 1u))) >> sh));
}

TK_API tk_fp16_t tk_float_to_fp16(float f);
TK_API tk_bf16_t tk_float_to_bf16(float f);
TK_API tk_t32_t  tk_float_to_t32(float f);
TK_API tk_f8_t   tk_float_to_f8(float f);
TK_API tk_e5m2_t tk_float_to_e5m2(float f);
TK_API tk_fp4_t  tk_float_to_fp4(float f);

/* ---- config-selected active type ------------------------------------------ */
#if TK_DTYPE == TK_DTYPE_FLOAT32
    typedef float tk_scalar_t;
    #define TK_TO_FLOAT(x)   ((float)(x))
    #define TK_FROM_FLOAT(f) ((float)(f))
    #define TK_DTYPE_NAME    "float32"
    #define TK_MANT_BITS     23
    #define TK_MIN_NORM_BEXP 1
    #define TK_SUB_SHIFT     149   /* float32: 2^-126 normal, 2^-149 subnormal step */
#elif TK_DTYPE == TK_DTYPE_FP16
    typedef tk_fp16_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_fp16_to_float(x)
    #define TK_FROM_FLOAT(f) tk__fp16_from_float(f)
    #define TK_DTYPE_NAME    "fp16"
    #define TK_MANT_BITS     10
    #define TK_MIN_NORM_BEXP 113
    #define TK_SUB_SHIFT     24   /* fp16: 2^-14 normal, 2^-24 subnormal step */
#elif TK_DTYPE == TK_DTYPE_BFLOAT16
    typedef tk_bf16_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_bf16_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_bf16(f)
    #define TK_DTYPE_NAME    "bfloat16"
    #define TK_MANT_BITS     7
    #define TK_MIN_NORM_BEXP 1
    #define TK_SUB_SHIFT     133   /* bf16: 2^-126 normal, 2^-133 subnormal step */
#elif TK_DTYPE == TK_DTYPE_TEKIN32
    typedef tk_t32_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_t32_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_t32(f)
    #define TK_DTYPE_NAME    "tekin32"
    #define TK_MANT_BITS     24
    #define TK_MIN_NORM_BEXP 65
    #define TK_SUB_SHIFT     86   /* tekin32: 2^-62 normal, 2^-86 subnormal step */
#elif TK_DTYPE == TK_DTYPE_TEKIN8
    typedef tk_f8_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_f8_to_float(x)
    #define TK_FROM_FLOAT(f) tk__f8_from_float(f)
    #define TK_DTYPE_NAME    "tekin8"
    #define TK_MANT_BITS     3
    #define TK_MIN_NORM_BEXP 121
    #define TK_SUB_SHIFT     9   /* tekin8 E4M3: 2^-6 normal, 2^-9 subnormal step */
#elif TK_DTYPE == TK_DTYPE_FP8_E5M2
    typedef tk_e5m2_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_e5m2_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_e5m2(f)
    #define TK_DTYPE_NAME    "fp8_e5m2"
    #define TK_MANT_BITS     2
    #define TK_MIN_NORM_BEXP 113
    #define TK_SUB_SHIFT     16   /* e5m2: 2^-14 normal, 2^-16 subnormal step */
#elif TK_DTYPE == TK_DTYPE_FP4_E2M1
    typedef tk_fp4_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_fp4_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_fp4(f)
    #define TK_DTYPE_NAME    "fp4_e2m1"
    #define TK_MANT_BITS     1
    #define TK_MIN_NORM_BEXP 127
    #define TK_SUB_SHIFT     1   /* fp4 E2M1: 1.0 is the smallest normal, 0.5 the only subnormal */
#else
    #error "Unknown TK_DTYPE in config.h"
#endif

TK_API const char *tk_dtype_name(void);   /* active storage type name */
TK_API int         tk_scalar_size(void);  /* sizeof(tk_scalar_t), for the binding */

#endif /* MANTISSA_DTYPES_H */
