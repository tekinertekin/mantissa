#ifndef MANTISSA_DTYPES_H
#define MANTISSA_DTYPES_H

#include <stdint.h>
#include <string.h>
#include "config.h"
#include "tk_export.h"

/* Narrow storage formats held in plain integers. See docs/DESIGN.md for the
 * bit layouts and the rationale behind tekin32 / tekin8.
 *
 *   fp16     1-5-10  IEEE-754 half   (Micikevicius et al., 2017)
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
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x3FFu;
    if (e == 0u)                                   /* zero or subnormal (m*2^-24) */
        return tk__u2f(s | tk__f2u((float)m * (1.0f / 16777216.0f)));
    if (e == 0x1Fu)                                /* inf / nan */
        return tk__u2f(s | 0x7F800000u | (m << 13));
    return tk__u2f(s | ((e + 112u) << 23) | (m << 13));   /* bias 15 -> 127 */
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

static inline float tk_f8_to_float(tk_f8_t v) {
    const uint32_t s = (v >> 7) & 1u, e = (v >> 3) & 0xFu, m = v & 7u;
    if (e == 0u) {                                 /* subnormal: m * 2^-9 */
        const float sub = (float)m * (1.0f / 512.0f);
        return s ? -sub : sub;
    }
    return tk__u2f((s << 31) | ((e + 120u) << 23) | (m << 20));  /* bias 7 -> 127 */
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

static inline float tk_fp4_to_float(tk_fp4_t v) {
    const uint32_t s = (v >> 3) & 1u, e = (v >> 1) & 3u, m = v & 1u;
    /* E2M1 has no inf/nan; the 8 magnitudes are 0,.5,1,1.5,2,3,4,6. */
    const float val = (e == 0u) ? (float)m * 0.5f
                                : (1.0f + 0.5f * (float)m) * (float)(1u << (e - 1u));
    return s ? -val : val;
}

/* ---- WRITE path: float32 -> narrow (cold; runs once at weight load) -------- */
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
#elif TK_DTYPE == TK_DTYPE_FP16
    typedef tk_fp16_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_fp16_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_fp16(f)
    #define TK_DTYPE_NAME    "fp16"
    #define TK_MANT_BITS     10
#elif TK_DTYPE == TK_DTYPE_BFLOAT16
    typedef tk_bf16_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_bf16_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_bf16(f)
    #define TK_DTYPE_NAME    "bfloat16"
    #define TK_MANT_BITS     7
#elif TK_DTYPE == TK_DTYPE_TEKIN32
    typedef tk_t32_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_t32_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_t32(f)
    #define TK_DTYPE_NAME    "tekin32"
    #define TK_MANT_BITS     24
#elif TK_DTYPE == TK_DTYPE_TEKIN8
    typedef tk_f8_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_f8_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_f8(f)
    #define TK_DTYPE_NAME    "tekin8"
    #define TK_MANT_BITS     3
#elif TK_DTYPE == TK_DTYPE_FP8_E5M2
    typedef tk_e5m2_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_e5m2_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_e5m2(f)
    #define TK_DTYPE_NAME    "fp8_e5m2"
    #define TK_MANT_BITS     2
#elif TK_DTYPE == TK_DTYPE_FP4_E2M1
    typedef tk_fp4_t tk_scalar_t;
    #define TK_TO_FLOAT(x)   tk_fp4_to_float(x)
    #define TK_FROM_FLOAT(f) tk_float_to_fp4(f)
    #define TK_DTYPE_NAME    "fp4_e2m1"
    #define TK_MANT_BITS     1
#else
    #error "Unknown TK_DTYPE in config.h"
#endif

TK_API const char *tk_dtype_name(void);   /* active storage type name */
TK_API int         tk_scalar_size(void);  /* sizeof(tk_scalar_t), for the binding */

#endif /* MANTISSA_DTYPES_H */
