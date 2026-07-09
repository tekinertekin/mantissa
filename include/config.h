#ifndef MANTISSA_CONFIG_H
#define MANTISSA_CONFIG_H

/* Build-time configuration. The main choice is the storage type used for
 * weights and activations. Math is always accumulated in float32 (see ops.c),
 * which is the standard mixed-precision recipe: narrow storage, wide accumulate
 * (Micikevicius et al., 2017, arXiv:1710.03740). */

#define TK_DTYPE_FLOAT32   0   /* IEEE-754 binary32          (1-8-23), 4 bytes */
#define TK_DTYPE_FP16      1   /* IEEE-754 binary16          (1-5-10), 2 bytes */
#define TK_DTYPE_BFLOAT16  2   /* Google bfloat16            (1-8-7),  2 bytes */
#define TK_DTYPE_TEKIN32   3   /* custom range-vs-precision  (1-7-24), 4 bytes */
#define TK_DTYPE_TEKIN8    4   /* FP8 E4M3                   (1-4-3),  1 byte  */
#define TK_DTYPE_FP8_E5M2  5   /* FP8 E5M2, range variant   (1-5-2),  1 byte  */
#define TK_DTYPE_FP4_E2M1  6   /* FP4 E2M1 (MXFP4 element)  (1-2-1),  4 bits  */

/* Zero-config default: Google's bfloat16. Safest general-purpose choice (full
 * float32 range, drop-in for training) -- what a user who never edits this file
 * gets. Override at build time: -DTK_DTYPE=<id>. */
#ifndef TK_DTYPE
#define TK_DTYPE TK_DTYPE_BFLOAT16
#endif

/* Whether layers add a bias term by default. The core API also accepts a NULL
 * bias pointer at runtime (per-layer control), so a bias-free attention
 * projection and a bias-using MLP layer coexist in one model regardless of this
 * flag. This only sets the default the examples follow. */
#ifndef TK_USE_BIAS
#define TK_USE_BIAS 1
#endif

#endif /* MANTISSA_CONFIG_H */
