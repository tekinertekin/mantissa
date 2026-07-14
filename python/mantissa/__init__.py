"""Thin ctypes binding over the mantissa shared library.

The library can be compiled for any storage dtype (see config.h; the default is
bfloat16). This binding stays dtype-agnostic by calling the float32 entry point
`tk_linear_forward_f32`: Python always passes plain float buffers, and the C
side quantizes them through whatever storage type it was built for. So the same
Python code exercises float32, bfloat16, fp8, ... just by rebuilding the lib.
"""
from __future__ import annotations

import ctypes
import sys
import weakref as _weakref
from array import array
from pathlib import Path

try:                                   # optional: enables a zero-copy fast path
    import numpy as _np
except ImportError:                    # binding works without it (plain lists)
    _np = None

__all__ = ["Mantissa", "Prepared", "Trainer", "Session",
           "IDENTITY", "STEP", "SIGN", "RELU", "SIGMOID", "TANH", "GELU"]

# Activation ids must match include/activations.h.
IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU = range(7)

_f32p = ctypes.POINTER(ctypes.c_float)
_i32p = ctypes.POINTER(ctypes.c_int32)


def _as_c_float(seq, n: int, name: str = "buffer"):
    """Expose `seq` (length n) to C as float32 without a per-element Python copy
    when possible. Returns (arg, writeback): `arg` is what to pass to the C call;
    if `writeback` is not None, call it after the C call to reflect in-place
    mutation back to `seq`.

    Zero-copy (mutation is seen by the caller automatically, writeback=None) for
    a C-contiguous float32 numpy.ndarray or an array('f'); a plain list/tuple is
    boxed into a ctypes buffer and, if the C call mutates it, copied back.
    A float64 / non-contiguous numpy array silently falls back to the copy path
    -- pass C-contiguous float32 for zero-copy."""
    size = seq.size if _np is not None and isinstance(seq, _np.ndarray) else len(seq)
    if size != n:
        raise ValueError(f"{name}: expected {n} float32 values, got {size}")
    if _np is not None and isinstance(seq, _np.ndarray) \
            and seq.dtype == _np.float32 and seq.flags["C_CONTIGUOUS"]:
        return seq.ctypes.data_as(_f32p), None
    if isinstance(seq, array) and seq.typecode == "f":
        return (ctypes.c_float * n).from_buffer(seq), None
    buf = (ctypes.c_float * n)(*seq)                # list/tuple/other: one copy
    return buf, (lambda: seq.__setitem__(slice(None), buf))


def _bind_c_float(seq, n: int, name: str):
    """Pre-bind `seq` (length n) as a float32 pointer for repeated C calls.
    Zero-copy only: a C-contiguous float32 numpy array or an array('f') —
    anything else raises, because a hidden copy would both cost time and break
    the in-place mutation the caller relies on across calls."""
    if _np is not None and isinstance(seq, _np.ndarray):
        if seq.size != n:
            raise ValueError(f"{name}: expected {n} float32 values, got {seq.size}")
        if seq.dtype == _np.float32 and seq.flags["C_CONTIGUOUS"]:
            return seq.ctypes.data_as(_f32p)
        raise TypeError(f"{name}: need a C-contiguous float32 array for a "
                        f"pre-bound pointer (got dtype={seq.dtype})")
    if isinstance(seq, array) and seq.typecode == "f":
        if len(seq) != n:
            raise ValueError(f"{name}: expected {n} float32 values, got {len(seq)}")
        return (ctypes.c_float * n).from_buffer(seq)
    raise TypeError(f"{name}: need a C-contiguous float32 numpy array or "
                    f"array('f') for a pre-bound pointer")


def _as_c_int32(seq, n: int, name: str = "order"):
    """Expose an index sequence (length n) to C as int32. Zero-copy for a
    C-contiguous int32 numpy array; anything else (int64 permutation, list,
    array('i')) is boxed into a small n*4-byte ctypes buffer -- still far
    cheaper than copying the data rows it indexes."""
    size = seq.size if _np is not None and isinstance(seq, _np.ndarray) else len(seq)
    if size != n:
        raise ValueError(f"{name}: expected {n} int32 indices, got {size}")
    if _np is not None and isinstance(seq, _np.ndarray) \
            and seq.dtype == _np.int32 and seq.flags["C_CONTIGUOUS"]:
        return seq.ctypes.data_as(_i32p)
    return (ctypes.c_int32 * n)(*(int(i) for i in seq))


def _library_path() -> str:
    ext = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
    here = Path(__file__).resolve().parent           # python/mantissa (or the installed package)
    root = here.parent.parent                        # repo root in a source checkout
    # Prefer the library bundled inside the installed package (wheel), then fall
    # back to a source-tree `make dist` / `make lib`.
    for lib in (here / f"libmantissa.{ext}",
                root / "dist" / f"libmantissa.{ext}",
                root / "build" / f"libmantissa.{ext}"):
        if lib.exists():
            return str(lib)
    raise FileNotFoundError(
        f"libmantissa.{ext} not found (bundled, dist/, or build/). "
        f"Build it:  make dist   (or reinstall the wheel)"
    )


class Mantissa:
    def __init__(self, path: str | None = None):
        self._lib = ctypes.CDLL(path or _library_path())

        self._lib.tk_dtype_name.restype = ctypes.c_char_p
        self._lib.tk_scalar_size.restype = ctypes.c_int

        # narrow-storage primitives for the resident-weights inference path
        # (Prepared): quantize float32 into the storage dtype, and the narrow
        # GEMV that consumes it. Buffers are opaque bytes (the storage width is
        # tk_scalar_size()), so these take void* rather than a typed pointer.
        self._lib.tk_quantize.restype = None
        self._lib.tk_quantize.argtypes = [_f32p, ctypes.c_void_p, ctypes.c_int]
        self._lib.tk_linear_forward.restype = None
        self._lib.tk_linear_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, _f32p,  # W, x, bias, y
            ctypes.c_int, ctypes.c_int, ctypes.c_int,                  # out_dim, in_dim, act
        ]

        self._lib.tk_linear_forward_f32.restype = None
        self._lib.tk_linear_forward_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, x, bias, y
            ctypes.c_int, ctypes.c_int,       # out_dim, in_dim
            ctypes.c_int,                     # activation
        ]

        # one float32 SGD training step (forward + backward + update); returns loss
        self._lib.tk_train_step_f32.restype = ctypes.c_float
        self._lib.tk_train_step_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, bias, x, target
            ctypes.c_int, ctypes.c_int,       # out_dim, in_dim
            ctypes.c_int, ctypes.c_float,     # activation, lr
        ]

        # a whole epoch of sequential SGD in one call; returns the mean loss
        self._lib.tk_train_epoch_f32.restype = ctypes.c_float
        self._lib.tk_train_epoch_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, bias, X, targets
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_samples, out_dim, in_dim
            ctypes.c_int, ctypes.c_float,     # activation, lr
        ]

        # epoch variant with an optional visit order (int32 permutation, no row
        # copies) and an optional in-epoch pre-update mistake count
        self._lib.tk_train_epoch_order_f32.restype = ctypes.c_float
        self._lib.tk_train_epoch_order_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, bias, X, targets
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_samples, out_dim, in_dim
            ctypes.c_int, ctypes.c_float,     # activation, lr
            _i32p, ctypes.POINTER(ctypes.c_int),  # order (or NULL), mistakes (or NULL)
        ]

        self._lib.tk_sgd_update_list_f32.restype = None
        self._lib.tk_sgd_update_list_f32.argtypes = [
            ctypes.POINTER(_f32p), ctypes.POINTER(_f32p),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_float,
        ]
        self._lib.tk_upsample2d_nearest_f32.restype = None
        self._lib.tk_upsample2d_nearest_f32.argtypes = [
            _f32p, _f32p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self._lib.tk_upsample2d_nearest_backward_f32.restype = None
        self._lib.tk_upsample2d_nearest_backward_f32.argtypes = \
            self._lib.tk_upsample2d_nearest_f32.argtypes
        self._lib.tk_loss.restype = ctypes.c_float
        self._lib.tk_loss.argtypes = [_f32p, _f32p, _f32p,
                                      ctypes.c_int, ctypes.c_int]

        # one epoch of the mistake-driven Rosenblatt rule; returns mistakes
        self._lib.tk_perceptron_epoch_f32.restype = ctypes.c_int
        self._lib.tk_perceptron_epoch_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, bias, X, targets
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_samples, out_dim, in_dim
            ctypes.c_float, _i32p,            # lr, order (or NULL)
        ]

        # CNN primitive family (conv.h; >= 0.2.1) — pure float32, NCHW.
        # Consumers feature-detect with hasattr(tk, "conv2d_forward").
        self._lib.tk_conv2d_out_dim.restype = ctypes.c_int
        self._lib.tk_conv2d_out_dim.argtypes = [ctypes.c_int] * 4

        self._lib.tk_conv2d_forward_f32.restype = None
        self._lib.tk_conv2d_forward_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p, _f32p,    # X, K, bias, Z, Y
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, in_c, in_h, in_w
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_c, kh, kw
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # stride, pad, act
        ]
        self._lib.tk_conv2d_backward_f32.restype = None
        self._lib.tk_conv2d_backward_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # X, K, Z, dY
            _f32p, _f32p, _f32p,                  # dK, db, dX
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, in_c, in_h, in_w
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # out_c, kh, kw
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # stride, pad, act
        ]
        self._lib.tk_maxpool2d_f32.restype = None
        self._lib.tk_maxpool2d_f32.argtypes = [
            _f32p, _f32p, _i32p,                  # X, Y, argmax
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, c, in_h, in_w
            ctypes.c_int, ctypes.c_int,           # pool, stride
        ]
        self._lib.tk_maxpool2d_backward_f32.restype = None
        self._lib.tk_maxpool2d_backward_f32.argtypes = [
            _f32p, _i32p, _f32p,                  # dY, argmax, dX
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, c, in_h, in_w
            ctypes.c_int, ctypes.c_int,           # out_h, out_w
        ]
        self._lib.tk_linear_forward_batch_f32.restype = None
        self._lib.tk_linear_forward_batch_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p, _f32p,    # W, X, bias, Z, Y
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, out_dim, in_dim, act
        ]
        self._lib.tk_linear_backward_batch_f32.restype = None
        self._lib.tk_linear_backward_batch_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, X, Z, dY
            _f32p, _f32p, _f32p,                  # dW, db, dX
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n, out_dim, in_dim, act
        ]
        self._lib.tk_softmax_xent_f32.restype = ctypes.c_float
        self._lib.tk_softmax_xent_f32.argtypes = [
            _f32p, _i32p, _f32p,                  # logits, labels, dlogits
            ctypes.c_int, ctypes.c_int,           # n, classes
        ]
        self._lib.tk_sgd_update_f32.restype = None
        self._lib.tk_sgd_update_f32.argtypes = [
            _f32p, _f32p, ctypes.c_int, ctypes.c_float,  # W, dW, n, lr
        ]

    @property
    def dtype(self) -> str:
        """Storage type the loaded library was compiled for (e.g. 'bfloat16')."""
        return self._lib.tk_dtype_name().decode()

    def linear_forward(self, W, x, bias, out_dim: int, in_dim: int, act: int,
                       out=None):
        """y = act(W @ x + bias). W is row-major out_dim x in_dim; bias may be None.

        Values are quantized through the library's storage dtype internally.
        Pass W/x/bias as float32 numpy arrays or array('f') to skip the per-
        element copy (see _as_c_float). For repeated calls, pass `out=` (a
        float32 numpy array or array('f') of out_dim) — the result is written
        into it directly and `out` is returned: no per-call output allocation,
        no per-element boxing of the result. Without `out`, returns a fresh list
        of out_dim floats."""
        Wc, _ = _as_c_float(W, out_dim * in_dim, "W")  # read-only inputs: no writeback
        xc, _ = _as_c_float(x, in_dim, "x")
        bc = _as_c_float(bias, out_dim, "bias")[0] if bias is not None else None
        if out is not None:
            yc, y_wb = _as_c_float(out, out_dim, "out")
            self._lib.tk_linear_forward_f32(Wc, xc, bc, yc, out_dim, in_dim, act)
            if y_wb:                          # `out` was a plain list: copy back
                y_wb()
            return out
        yc = (ctypes.c_float * out_dim)()
        self._lib.tk_linear_forward_f32(Wc, xc, bc, yc, out_dim, in_dim, act)
        return list(yc)

    def prepare(self, W, out_dim: int, in_dim: int, bias=None) -> "Prepared":
        """Pre-quantize a dense layer's weights into the storage dtype and hold
        them resident narrow, for repeated inference on fixed weights. The
        returned object's forward() skips linear_forward's per-call
        re-quantization of every weight (linear_forward requantizes W on each
        call) and runs the narrow SIMD kernel directly — measured 1.9x at 16x16
        up to ~35x at 1024x1024 in Python (bf16, serial), at half the weight
        bytes of float32. W is row-major out_dim x in_dim; bias may be None."""
        return Prepared(self._lib, W, out_dim, in_dim, bias)

    def train_step(self, W, x, target, out_dim: int, in_dim: int,
                   act: int, lr: float, bias=None) -> float:
        """One SGD step on a dense layer. Mutates W (and bias) in place, returns
        the MSE loss before the update. W is row-major out_dim x in_dim.

        Pass W/bias as float32 numpy arrays or array('f') for a zero-copy update
        (mutated directly, no list round-trip); a plain list is copied in and
        copied back."""
        Wc, W_wb = _as_c_float(W, out_dim * in_dim, "W")   # mutated -> may need writeback
        xc, _ = _as_c_float(x, in_dim, "x")
        tc, _ = _as_c_float(target, out_dim, "target")
        bc, b_wb = _as_c_float(bias, out_dim, "bias") if bias is not None else (None, None)
        loss = self._lib.tk_train_step_f32(Wc, bc, xc, tc, out_dim, in_dim, act, lr)
        if W_wb:                              # reflect update back for the copy path
            W_wb()
        if b_wb:
            b_wb()
        return float(loss)

    def train_epoch(self, W, X, targets, n_samples: int, out_dim: int,
                    in_dim: int, act: int, lr: float, bias=None,
                    order=None, mistakes: bool = False):
        """A full epoch of sequential SGD in ONE C call: X is n_samples rows of
        in_dim, targets n_samples rows of out_dim (both flat, row-major).
        Weight updates are numerically identical to calling train_step once per
        sample; the win is one FFI crossing per epoch instead of one per sample.
        Mutates W (and bias) in place; returns the mean pre-update loss.

        order: optional visit sequence (a permutation of range(n_samples)).
        Pass a C-contiguous int32 numpy array for zero-copy (e.g.
        `rng.permutation(n).astype(np.int32)`) -- this replaces materializing
        row-permuted copies of X/targets each epoch, with a bit-identical
        weight trajectory for the same sequence.

        mistakes=True additionally returns the in-epoch mistake count -- the
        number of (sample, output-row) pairs whose PRE-update linear response
        disagreed with the target's sign (target*z <= 0) as the epoch visited
        them -- as a (loss, mistakes) tuple, at no extra pass over the data.
        Note: an in-epoch pre-update count is not the same number as a
        post-epoch pass over the data with the final weights."""
        Wc, W_wb = _as_c_float(W, out_dim * in_dim, "W")
        Xc, _ = _as_c_float(X, n_samples * in_dim, "X")
        tc, _ = _as_c_float(targets, n_samples * out_dim, "targets")
        bc, b_wb = _as_c_float(bias, out_dim, "bias") if bias is not None else (None, None)
        if order is None and not mistakes:
            loss = self._lib.tk_train_epoch_f32(Wc, bc, Xc, tc,
                                                n_samples, out_dim, in_dim, act, lr)
            m = None
        else:
            oc = _as_c_int32(order, n_samples) if order is not None else None
            mc = ctypes.c_int(0) if mistakes else None
            loss = self._lib.tk_train_epoch_order_f32(
                Wc, bc, Xc, tc, n_samples, out_dim, in_dim, act, lr,
                oc, ctypes.byref(mc) if mistakes else None)
            m = mc.value if mistakes else None
        if W_wb:
            W_wb()
        if b_wb:
            b_wb()
        return (float(loss), m) if mistakes else float(loss)

    def trainer(self, W, X, targets, n_samples: int, out_dim: int,
                in_dim: int, bias=None) -> "Trainer":
        """Pre-bind a training session over fixed buffers, for the
        epoch-in-a-loop pattern. Measured (Apple M4, 1030x4 float32): one
        perceptron_epoch wrapper call costs ~9.8 us of which only ~3 us is the
        C epoch — the rest is re-deriving ctypes pointers for the same five
        unchanged buffers every epoch. Trainer derives them once; per-epoch
        calls then pass only lr and the visit order. All buffers must be
        C-contiguous float32 (numpy or array('f')) — they are mutated in place
        and must stay alive for the Trainer's lifetime."""
        return Trainer(self._lib, W, X, targets, n_samples, out_dim, in_dim, bias)

    def perceptron_epoch(self, W, X, targets, n_samples: int, out_dim: int,
                         in_dim: int, lr: float, bias=None, order=None) -> int:
        """One epoch of the mistake-driven perceptron rule (Rosenblatt, 1958)
        in ONE C call, plain float32: targets are +-1 (n_samples rows of
        out_dim); per visited sample and row, z = w.x + b, and on a mistake
        (target*z <= 0 -- a zero margin counts) w += lr*target*x,
        b += lr*target. Mutates W (and bias) in place; returns the number of
        mistakes this epoch (0 = the data was separated this pass). `order` as
        in train_epoch. Replaces a per-sample forward + numpy update loop with
        one FFI crossing per epoch."""
        Wc, W_wb = _as_c_float(W, out_dim * in_dim, "W")
        Xc, _ = _as_c_float(X, n_samples * in_dim, "X")
        tc, _ = _as_c_float(targets, n_samples * out_dim, "targets")
        bc, b_wb = _as_c_float(bias, out_dim, "bias") if bias is not None else (None, None)
        oc = _as_c_int32(order, n_samples) if order is not None else None
        m = self._lib.tk_perceptron_epoch_f32(Wc, bc, Xc, tc,
                                              n_samples, out_dim, in_dim, lr, oc)
        if W_wb:
            W_wb()
        if b_wb:
            b_wb()
        return int(m)

    # ---- CNN primitives (pure float32, NCHW, batch outermost) --------------
    # All buffers are flat, C-contiguous float32 (argmax/labels: int32) numpy
    # arrays for zero-copy; outputs are mutated in place. Feature-detect with
    # hasattr(tk, "conv2d_forward") (needs mantissa-nn >= 0.2.1).

    def conv2d_out_dim(self, in_dim: int, k: int, stride: int, pad: int) -> int:
        """Conv/pool output spatial size: (in + 2*pad - k)//stride + 1,
        clamped to >= 0 (0 on degenerate arguments).

        Computed in Python, mirroring tk_conv2d_out_dim BIT-EXACTLY — including
        C's truncate-toward-zero division, which for the degenerate band
        -stride < in+2p-k < 0 yields 1, not 0 (Python floor-div would differ;
        a 152-case grid cross-check against the C export caught exactly that).
        The kernel computes oh/ow with the same C function internally, so the
        binding MUST agree or buffer sizes go wrong. The C export stays the
        ABI source of truth (tests/test_edges.c pins its table); the FFI
        round-trip for this pure integer formula measured 50% of ALL crossings
        in a downstream CNN fit (2 per conv/pool call)."""
        if in_dim <= 0 or k <= 0 or stride <= 0 or pad < 0:
            return 0
        span = in_dim + 2 * pad - k
        q = span // stride if span >= 0 else -((-span) // stride)  # C trunc div
        out = q + 1
        return out if out > 0 else 0

    def conv2d_forward(self, X, K, bias, Z, Y, n: int, in_c: int, in_h: int,
                       in_w: int, out_c: int, kh: int, kw: int,
                       stride: int, pad: int, act: int):
        """Batched conv2d forward: Y = act(conv(X, K) + bias), NCHW.
        X: n*in_c*in_h*in_w, K: out_c*in_c*kh*kw, bias: out_c or None,
        Z (pre-activation): n*out_c*oh*ow, or None for ANY activation when
        no backward pass will follow — inference then skips a full
        output-sized store per layer (backward still requires the saved Z
        for non-identity activations; for identity, Y works: Z == Y there).
        Y: same shape as Z. Returns Y. im2col + GEMM in C,
        one sample's patch matrix at a time, threaded over the batch.
        Measured (M4, VGG 64x32x32->64@3x3 pad 1, batch 16): 24.8 ms serial /
        5.97 ms threaded, 202 GFLOP/s."""
        oh = self.conv2d_out_dim(in_h, kh, stride, pad)
        ow = self.conv2d_out_dim(in_w, kw, stride, pad)
        ysz = n * out_c * oh * ow
        Xc, _ = _as_c_float(X, n * in_c * in_h * in_w, "X")
        Kc, _ = _as_c_float(K, out_c * in_c * kh * kw, "K")
        bc = _as_c_float(bias, out_c, "bias")[0] if bias is not None else None
        Zc, Z_wb = _as_c_float(Z, ysz, "Z") if Z is not None else (None, None)
        Yc, Y_wb = _as_c_float(Y, ysz, "Y")
        self._lib.tk_conv2d_forward_f32(Xc, Kc, bc, Zc, Yc, n, in_c, in_h,
                                        in_w, out_c, kh, kw, stride, pad, act)
        if Z_wb:
            Z_wb()
        if Y_wb:
            Y_wb()
        return Y

    def conv2d_backward(self, X, K, Z, dY, dK, db, dX, n: int, in_c: int,
                        in_h: int, in_w: int, out_c: int, kh: int, kw: int,
                        stride: int, pad: int, act: int):
        """Batched conv2d backward. Writes dK (out_c*in_c*kh*kw, summed over
        the batch), db (out_c, summed, or None) and dX (n*in_c*in_h*in_w, or
        None for the first layer); Z is the saved pre-activation. dK via
        im2col^T accumulation, dX via col2im scatter. Measured (M4, VGG
        64x32x32->64@3x3 pad 1, batch 16): 54.7 ms serial / 16.6 threaded."""
        oh = self.conv2d_out_dim(in_h, kh, stride, pad)
        ow = self.conv2d_out_dim(in_w, kw, stride, pad)
        ysz = n * out_c * oh * ow
        xsz = n * in_c * in_h * in_w
        ksz = out_c * in_c * kh * kw
        Xc, _ = _as_c_float(X, xsz, "X")
        Kc, _ = _as_c_float(K, ksz, "K")
        Zc, _ = _as_c_float(Z, ysz, "Z")
        dYc, _ = _as_c_float(dY, ysz, "dY")
        dKc, dK_wb = _as_c_float(dK, ksz, "dK")
        dbc, db_wb = _as_c_float(db, out_c, "db") if db is not None else (None, None)
        dXc, dX_wb = _as_c_float(dX, xsz, "dX") if dX is not None else (None, None)
        self._lib.tk_conv2d_backward_f32(Xc, Kc, Zc, dYc, dKc, dbc, dXc,
                                         n, in_c, in_h, in_w, out_c, kh, kw,
                                         stride, pad, act)
        for wb in (dK_wb, db_wb, dX_wb):
            if wb:
                wb()

    def maxpool2d(self, X, Y, argmax, n: int, c: int, in_h: int, in_w: int,
                  pool: int, stride: int):
        """Max pooling, no padding, floor semantics: oh = (in_h - pool)//stride
        + 1 (ragged edges dropped). Y: n*c*oh*ow float32; argmax: same shape,
        int32 (pass a numpy int32 array — it receives each winner's flat
        h*in_w + w index, which maxpool2d_backward consumes). Returns Y."""
        oh = self.conv2d_out_dim(in_h, pool, stride, 0)
        ow = self.conv2d_out_dim(in_w, pool, stride, 0)
        ysz = n * c * oh * ow
        Xc, _ = _as_c_float(X, n * c * in_h * in_w, "X")
        Yc, Y_wb = _as_c_float(Y, ysz, "Y")
        # argmax is an OUTPUT: _as_c_int32's boxed fallback for non-int32
        # input would be written by C and silently discarded (no writeback)
        # — a later maxpool2d_backward through it would scatter garbage.
        # Found by the binding contract tests; require zero-copy here.
        if _np is not None and isinstance(argmax, _np.ndarray) \
                and argmax.dtype != _np.int32:
            raise TypeError(f"argmax: needs an int32 numpy array (it is "
                            f"written by C); got dtype={argmax.dtype}")
        ac = _as_c_int32(argmax, ysz, "argmax")
        self._lib.tk_maxpool2d_f32(Xc, Yc, ac, n, c, in_h, in_w, pool, stride)
        if Y_wb:
            Y_wb()
        return Y

    def maxpool2d_backward(self, dY, argmax, dX, n: int, c: int, in_h: int,
                           in_w: int, out_h: int, out_w: int):
        """Max pooling backward: zeroes dX (n*c*in_h*in_w) then scatters dY
        (n*c*out_h*out_w) through argmax. Overlapping windows accumulate."""
        ysz = n * c * out_h * out_w
        dYc, _ = _as_c_float(dY, ysz, "dY")
        ac = _as_c_int32(argmax, ysz, "argmax")
        dXc, dX_wb = _as_c_float(dX, n * c * in_h * in_w, "dX")
        self._lib.tk_maxpool2d_backward_f32(dYc, ac, dXc, n, c, in_h, in_w,
                                            out_h, out_w)
        if dX_wb:
            dX_wb()

    def linear_forward_batch(self, W, X, bias, Z, Y, n: int, out_dim: int,
                             in_dim: int, act: int):
        """Batched float32 dense forward (the CNN head): Y = act(X @ W^T +
        bias). X: n*in_dim, W: out_dim*in_dim row-major, Z/Y: n*out_dim
        (Z may be None). Returns Y. Same 4-row register-blocked kernel as the
        conv GEMM."""
        Wc, _ = _as_c_float(W, out_dim * in_dim, "W")
        Xc, _ = _as_c_float(X, n * in_dim, "X")
        bc = _as_c_float(bias, out_dim, "bias")[0] if bias is not None else None
        Zc, Z_wb = _as_c_float(Z, n * out_dim, "Z") if Z is not None else (None, None)
        Yc, Y_wb = _as_c_float(Y, n * out_dim, "Y")
        self._lib.tk_linear_forward_batch_f32(Wc, Xc, bc, Zc, Yc,
                                              n, out_dim, in_dim, act)
        if Z_wb:
            Z_wb()
        if Y_wb:
            Y_wb()
        return Y

    def linear_backward_batch(self, W, X, Z, dY, dW, db, dX, n: int,
                              out_dim: int, in_dim: int, act: int):
        """Batched float32 dense backward: writes dW (out_dim*in_dim, summed
        over the batch), db (out_dim, summed, or None) and dX (n*in_dim, or
        None); Z is the saved pre-activation, dz = dy * act'(z)."""
        Wc, _ = _as_c_float(W, out_dim * in_dim, "W")
        Xc, _ = _as_c_float(X, n * in_dim, "X")
        Zc, _ = _as_c_float(Z, n * out_dim, "Z")
        dYc, _ = _as_c_float(dY, n * out_dim, "dY")
        dWc, dW_wb = _as_c_float(dW, out_dim * in_dim, "dW")
        dbc, db_wb = _as_c_float(db, out_dim, "db") if db is not None else (None, None)
        dXc, dX_wb = _as_c_float(dX, n * in_dim, "dX") if dX is not None else (None, None)
        self._lib.tk_linear_backward_batch_f32(Wc, Xc, Zc, dYc, dWc, dbc, dXc,
                                               n, out_dim, in_dim, act)
        for wb in (dW_wb, db_wb, dX_wb):
            if wb:
                wb()

    def softmax_xent(self, logits, labels, dlogits, n: int, classes: int) -> float:
        """Fused softmax + cross-entropy, numerically stable (row-max
        subtraction): logits n*classes float32, labels n int32 class ids.
        Writes dlogits = (softmax - onehot)/n and returns the mean loss."""
        lc, _ = _as_c_float(logits, n * classes, "logits")
        yc = _as_c_int32(labels, n, "labels")
        dc, d_wb = _as_c_float(dlogits, n * classes, "dlogits")
        loss = self._lib.tk_softmax_xent_f32(lc, yc, dc, n, classes)
        if d_wb:
            d_wb()
        return float(loss)

    def sgd_update(self, W, dW, n: int, lr: float):
        """Plain float32 SGD: W -= lr * dW over n elements, in place (the
        narrow-storage counterpart is tk_sgd_step; the CNN family trains
        pure float32)."""
        Wc, W_wb = _as_c_float(W, n, "W")
        dWc, _ = _as_c_float(dW, n, "dW")
        self._lib.tk_sgd_update_f32(Wc, dWc, n, lr)
        if W_wb:
            W_wb()

    def upsample2d(self, X, Y, n: int, c: int, in_h: int, in_w: int, k: int):
        """Nearest-neighbor upsample by integer factor k (NCHW): X
        n*c*in_h*in_w -> Y n*c*(in_h*k)*(in_w*k). Returns Y. The decoder op
        of upsample+conv architectures (Odena, Dumoulin & Olah, 2016).
        Measured vs the numpy broadcast-assign at autoencoder shapes
        (batch 32): 200 -> 41 us and 416 -> 37 us."""
        Xc, _ = _as_c_float(X, n * c * in_h * in_w, "X")
        Yc, Y_wb = _as_c_float(Y, n * c * in_h * k * in_w * k, "Y")
        self._lib.tk_upsample2d_nearest_f32(Xc, Yc, n, c, in_h, in_w, k)
        if Y_wb:
            Y_wb()
        return Y

    def upsample2d_backward(self, dY, dX, n: int, c: int, in_h: int,
                            in_w: int, k: int):
        """Exact adjoint of upsample2d: dX = k x k block-sum of dY
        (dY n*c*(in_h*k)*(in_w*k) -> dX n*c*in_h*in_w). Measured vs numpy's
        fused np.sum(axis=(3,5)): 739 -> 21 us and 1415 -> 30 us (35-47x —
        the reduction iterator degenerates on interleaved length-k axes)."""
        dYc, _ = _as_c_float(dY, n * c * in_h * k * in_w * k, "dY")
        dXc, dX_wb = _as_c_float(dX, n * c * in_h * in_w, "dX")
        self._lib.tk_upsample2d_nearest_backward_f32(dYc, dXc, n, c,
                                                     in_h, in_w, k)
        if dX_wb:
            dX_wb()
        return dX

    def sgd_update_list(self, Ws, dWs, lr: float):
        """SGD over a LIST of parameter tensors in ONE crossing:
        Ws[i] -= lr * dWs[i], elementwise, C-contiguous float32 numpy arrays
        only (in-place mutation is the point). Bit-identical to calling
        sgd_update per tensor; a 16-float bias update costs more to cross
        than to compute, and a small model updates 8-12 tensors per batch."""
        count = len(Ws)
        if len(dWs) != count:
            raise ValueError(f"Ws has {count} tensors but dWs has {len(dWs)}")
        wp = (_f32p * count)()
        dp = (_f32p * count)()
        ns = (ctypes.c_int * count)()
        for i, (w, d) in enumerate(zip(Ws, dWs)):
            if d.size != w.size:
                raise ValueError(f"tensor {i}: W has {w.size} elements, "
                                 f"dW has {d.size}")
            wp[i] = _bind_c_float(w, w.size, f"Ws[{i}]")
            dp[i] = _bind_c_float(d, d.size, f"dWs[{i}]")
            ns[i] = w.size
        self._lib.tk_sgd_update_list_f32(wp, dp, ns, count, lr)

    def mse_loss(self, Y, T, dY, n: int) -> float:
        """Fused MSE loss + seed gradient over flat length-n float32 buffers
        (tk_loss, TK_LOSS_MSE): writes dY = 2*(Y-T)/n, returns mean((Y-T)^2).
        Exposed for API completeness — the C loop is scalar and memory-bound,
        so a well-formed numpy expression is not slower; use whichever keeps
        your loop's buffers stable."""
        Yc, _ = _as_c_float(Y, n, "Y")
        Tc, _ = _as_c_float(T, n, "T")
        dc, d_wb = _as_c_float(dY, n, "dY")
        loss = self._lib.tk_loss(Yc, Tc, dc, n, 0)
        if d_wb:
            d_wb()
        return float(loss)

    def session(self) -> "Session":
        """An identity-memoized view of the CNN-primitive methods for
        training loops over fixed buffers — see Session."""
        return Session(self)


class Session:
    """The CNN-primitive methods with identity-memoized pointers: same
    signatures, same semantics, but each distinct array's ctypes pointer is
    derived once and reused for as long as the same object keeps arriving
    (a memo hit is a dict lookup plus an `is` check, not a numpy `.ctypes`
    walk — the walk costs ~1.3 µs and the Trainer work showed it dominating
    small-op calls).

    Built for the mantissa-cnn training-loop shape: parameters, scratch and
    staging buffers are allocated once per fit and refilled in place, so
    after the first batch every call here is pure FFI. Measured on the
    LeNet-5/MNIST fit (M4): pointer conversion was ~12% of wall time
    (16.8k conversions); a Session removes almost all of it.

    The memo references its arrays WEAKLY: a cached entry is used only while
    the exact same object is still alive (checked by identity on every hit),
    and arrays the caller drops are not kept alive by the Session. The first
    version held strong references — measured downstream, that retained every
    fresh inference slice and per-batch noise array for the model's lifetime
    (~100 KB per predict call / per denoise batch, unbounded). Buffers must
    be C-contiguous numpy arrays (float32; int32 for argmax/labels): the
    zero-copy requirement is the point, so anything else raises instead of
    silently copying. Volatile arrays (a fresh object per call) still work —
    they simply pay the one-time pointer walk each call; reuse a staging
    buffer to get memo hits."""

    _SWEEP_AT = 4096   # purge dead weakref entries when the memo grows past this

    def __init__(self, tk: "Mantissa"):
        self._tk = tk
        self._lib = tk._lib
        self._fmemo = {}
        self._imemo = {}
        self._lmemo = {}   # sgd_update_list pointer-array memo

    @staticmethod
    def _memo_get(memo, arr):
        hit = memo.get(id(arr))
        if hit is not None and hit[0]() is arr:
            return hit[1]
        return None

    def _memo_put(self, memo, arr, p):
        # id() keys stay valid only while the object lives; the identity check
        # in _memo_get makes a recycled id at worst a rebind, never a stale
        # pointer. Dead entries are swept in bulk so the dict stays bounded
        # even under a fresh-array-per-call pattern.
        if len(memo) >= self._SWEEP_AT:
            for k in [k for k, v in memo.items() if v[0]() is None]:
                del memo[k]
        try:
            memo[id(arr)] = (_weakref.ref(arr), p)
        except TypeError:                     # non-weakrefable (array('f'))
            pass                              # correct, just unmemoized
        return p

    def _fp(self, arr, n: int, name: str):
        p = self._memo_get(self._fmemo, arr)
        if p is not None:
            return p
        _bind_c_float(arr, n, name)           # validation (raises on misuse)
        if not isinstance(arr, _np.ndarray):  # array('f'): valid but unmemoized
            return _bind_c_float(arr, n, name)
        # Pointer built from the raw address, NOT data_as(): numpy's data_as
        # embeds a strong reference to the array inside the pointer object
        # (its use-after-free guard), which would re-pin every array through
        # the weakref memo — measured: 50 volatile slices stayed alive. The
        # caller's own reference keeps the array valid for the duration of
        # each C call; between calls the weakref identity check protects us.
        return self._memo_put(self._fmemo, arr,
                              ctypes.cast(arr.ctypes.data, _f32p))

    def _ip(self, arr, n: int, name: str):
        p = self._memo_get(self._imemo, arr)
        if p is not None:
            return p
        if not (_np is not None and isinstance(arr, _np.ndarray)
                and arr.dtype == _np.int32 and arr.flags["C_CONTIGUOUS"]):
            raise TypeError(f"{name}: Session needs a C-contiguous int32 "
                            f"numpy array")
        if arr.size != n:
            raise ValueError(f"{name}: expected {n} int32 values, got {arr.size}")
        return self._memo_put(self._imemo, arr,
                              ctypes.cast(arr.ctypes.data, _i32p))

    def conv2d_out_dim(self, in_dim: int, k: int, stride: int, pad: int) -> int:
        return self._tk.conv2d_out_dim(in_dim, k, stride, pad)

    def conv2d_forward(self, X, K, bias, Z, Y, n, in_c, in_h, in_w,
                       out_c, kh, kw, stride, pad, act):
        oh = self.conv2d_out_dim(in_h, kh, stride, pad)
        ow = self.conv2d_out_dim(in_w, kw, stride, pad)
        ysz = n * out_c * oh * ow
        self._lib.tk_conv2d_forward_f32(
            self._fp(X, n * in_c * in_h * in_w, "X"),
            self._fp(K, out_c * in_c * kh * kw, "K"),
            self._fp(bias, out_c, "bias") if bias is not None else None,
            self._fp(Z, ysz, "Z") if Z is not None else None,
            self._fp(Y, ysz, "Y"),
            n, in_c, in_h, in_w, out_c, kh, kw, stride, pad, act)
        return Y

    def conv2d_backward(self, X, K, Z, dY, dK, db, dX, n, in_c, in_h, in_w,
                        out_c, kh, kw, stride, pad, act):
        oh = self.conv2d_out_dim(in_h, kh, stride, pad)
        ow = self.conv2d_out_dim(in_w, kw, stride, pad)
        ysz = n * out_c * oh * ow
        xsz = n * in_c * in_h * in_w
        ksz = out_c * in_c * kh * kw
        self._lib.tk_conv2d_backward_f32(
            self._fp(X, xsz, "X"), self._fp(K, ksz, "K"),
            self._fp(Z, ysz, "Z"), self._fp(dY, ysz, "dY"),
            self._fp(dK, ksz, "dK"),
            self._fp(db, out_c, "db") if db is not None else None,
            self._fp(dX, xsz, "dX") if dX is not None else None,
            n, in_c, in_h, in_w, out_c, kh, kw, stride, pad, act)

    def maxpool2d(self, X, Y, argmax, n, c, in_h, in_w, pool, stride):
        oh = self.conv2d_out_dim(in_h, pool, stride, 0)
        ow = self.conv2d_out_dim(in_w, pool, stride, 0)
        ysz = n * c * oh * ow
        self._lib.tk_maxpool2d_f32(
            self._fp(X, n * c * in_h * in_w, "X"), self._fp(Y, ysz, "Y"),
            self._ip(argmax, ysz, "argmax"), n, c, in_h, in_w, pool, stride)
        return Y

    def maxpool2d_backward(self, dY, argmax, dX, n, c, in_h, in_w,
                           out_h, out_w):
        ysz = n * c * out_h * out_w
        self._lib.tk_maxpool2d_backward_f32(
            self._fp(dY, ysz, "dY"), self._ip(argmax, ysz, "argmax"),
            self._fp(dX, n * c * in_h * in_w, "dX"),
            n, c, in_h, in_w, out_h, out_w)

    def linear_forward_batch(self, W, X, bias, Z, Y, n, out_dim, in_dim, act):
        self._lib.tk_linear_forward_batch_f32(
            self._fp(W, out_dim * in_dim, "W"), self._fp(X, n * in_dim, "X"),
            self._fp(bias, out_dim, "bias") if bias is not None else None,
            self._fp(Z, n * out_dim, "Z") if Z is not None else None,
            self._fp(Y, n * out_dim, "Y"), n, out_dim, in_dim, act)
        return Y

    def linear_backward_batch(self, W, X, Z, dY, dW, db, dX, n,
                              out_dim, in_dim, act):
        self._lib.tk_linear_backward_batch_f32(
            self._fp(W, out_dim * in_dim, "W"), self._fp(X, n * in_dim, "X"),
            self._fp(Z, n * out_dim, "Z"), self._fp(dY, n * out_dim, "dY"),
            self._fp(dW, out_dim * in_dim, "dW"),
            self._fp(db, out_dim, "db") if db is not None else None,
            self._fp(dX, n * in_dim, "dX") if dX is not None else None,
            n, out_dim, in_dim, act)

    def softmax_xent(self, logits, labels, dlogits, n, classes) -> float:
        loss = self._lib.tk_softmax_xent_f32(
            self._fp(logits, n * classes, "logits"),
            self._ip(labels, n, "labels"),
            self._fp(dlogits, n * classes, "dlogits"), n, classes)
        return float(loss)

    def sgd_update(self, W, dW, n, lr):
        self._lib.tk_sgd_update_f32(self._fp(W, n, "W"),
                                    self._fp(dW, n, "dW"), n, lr)

    def upsample2d(self, X, Y, n, c, in_h, in_w, k):
        self._lib.tk_upsample2d_nearest_f32(
            self._fp(X, n * c * in_h * in_w, "X"),
            self._fp(Y, n * c * in_h * k * in_w * k, "Y"),
            n, c, in_h, in_w, k)
        return Y

    def upsample2d_backward(self, dY, dX, n, c, in_h, in_w, k):
        self._lib.tk_upsample2d_nearest_backward_f32(
            self._fp(dY, n * c * in_h * k * in_w * k, "dY"),
            self._fp(dX, n * c * in_h * in_w, "dX"),
            n, c, in_h, in_w, k)
        return dX

    def sgd_update_list(self, Ws, dWs, lr):
        """One crossing for the whole parameter list. The pointer arrays are
        memoized per Ws-list identity, and a hit re-verifies the identity of
        EVERY tensor (weakref + `is`, ~60 ns each) — replacing a tensor in
        the list triggers a clean rebuild, never a stale pointer."""
        hit = self._lmemo.get(id(Ws))
        if hit is not None:
            refs, args = hit
            if len(refs) == len(Ws) + len(dWs) and all(
                    r() is t for r, t in zip(refs, Ws)) and all(
                    r() is t for r, t in zip(refs[len(Ws):], dWs)):
                self._lib.tk_sgd_update_list_f32(*args, lr)
                return
        count = len(Ws)
        if len(dWs) != count:
            raise ValueError(f"Ws has {count} tensors but dWs has {len(dWs)}")
        wp = (_f32p * count)()
        dp = (_f32p * count)()
        ns = (ctypes.c_int * count)()
        for i, (w, d) in enumerate(zip(Ws, dWs)):
            if d.size != w.size:
                raise ValueError(f"tensor {i}: W has {w.size} elements, "
                                 f"dW has {d.size}")
            wp[i] = ctypes.cast(self._fp(w, w.size, f"Ws[{i}]"), _f32p)
            dp[i] = ctypes.cast(self._fp(d, d.size, f"dWs[{i}]"), _f32p)
            ns[i] = w.size
        try:
            refs = tuple(_weakref.ref(t) for t in list(Ws) + list(dWs))
            self._lmemo[id(Ws)] = (refs, (wp, dp, ns, count))
        except TypeError:
            pass
        self._lib.tk_sgd_update_list_f32(wp, dp, ns, count, lr)

    def mse_loss(self, Y, T, dY, n) -> float:
        return float(self._lib.tk_loss(self._fp(Y, n, "Y"),
                                       self._fp(T, n, "T"),
                                       self._fp(dY, n, "dY"), n, 0))


class Trainer:
    """A training session with pre-bound pointers: W (out_dim x in_dim), X
    (n_samples x in_dim), targets (n_samples x out_dim) and optional bias are
    converted to C pointers ONCE, so each epoch call crosses the FFI with no
    per-buffer conversion work. Construct via Mantissa.trainer().

    Semantics are identical to the corresponding Mantissa methods — same C
    entry points, same buffers, bit-identical weight trajectories; only the
    per-call Python overhead is gone. W and bias are mutated in place by the
    epoch calls, exactly as with the un-bound methods.

    The Trainer holds references to the arrays it binds, but the pointers are
    to the arrays' CURRENT storage: do not resize or reassign them mid-session.
    """

    def __init__(self, lib, W, X, targets, n_samples: int, out_dim: int,
                 in_dim: int, bias=None):
        self._lib = lib
        self.n_samples, self.out_dim, self.in_dim = n_samples, out_dim, in_dim
        self._Wp = _bind_c_float(W, out_dim * in_dim, "W")
        self._Xp = _bind_c_float(X, n_samples * in_dim, "X")
        self._tp = _bind_c_float(targets, n_samples * out_dim, "targets")
        self._bp = _bind_c_float(bias, out_dim, "bias") if bias is not None else None
        self._refs = (W, X, targets, bias)            # keep the storage alive

    def perceptron_epoch(self, lr: float, order=None) -> int:
        """One mistake-driven epoch (see Mantissa.perceptron_epoch); returns
        the mistake count. `order` as in train_epoch."""
        oc = _as_c_int32(order, self.n_samples) if order is not None else None
        return int(self._lib.tk_perceptron_epoch_f32(
            self._Wp, self._bp, self._Xp, self._tp,
            self.n_samples, self.out_dim, self.in_dim, lr, oc))

    def train_epoch(self, act: int, lr: float, order=None, mistakes: bool = False):
        """One epoch of sequential SGD (see Mantissa.train_epoch); returns the
        mean pre-update loss, or (loss, in-epoch mistakes) with mistakes=True."""
        if order is None and not mistakes:
            return float(self._lib.tk_train_epoch_f32(
                self._Wp, self._bp, self._Xp, self._tp,
                self.n_samples, self.out_dim, self.in_dim, act, lr))
        oc = _as_c_int32(order, self.n_samples) if order is not None else None
        mc = ctypes.c_int(0) if mistakes else None
        loss = self._lib.tk_train_epoch_order_f32(
            self._Wp, self._bp, self._Xp, self._tp,
            self.n_samples, self.out_dim, self.in_dim, act, lr,
            oc, ctypes.byref(mc) if mistakes else None)
        return (float(loss), mc.value) if mistakes else float(loss)

    def margins(self, out):
        """Linear responses z_s = w . x_s for every sample, WITHOUT bias, into
        `out` (C-contiguous float32, length n_samples; returned). Single-output
        layers only: with out_dim == 1 the whole batch is one row-parallel GEMV
        (X as the weight matrix, w as the input). The caller adds its scalar
        bias — one vectorized add — and keeps full control of the comparison
        (the post-epoch convergence count this exists for)."""
        if self.out_dim != 1:
            raise ValueError(f"margins() needs out_dim == 1, got {self.out_dim}")
        yp = _bind_c_float(out, self.n_samples, "out")
        self._lib.tk_linear_forward_f32(self._Xp, self._Wp, None, yp,
                                        self.n_samples, self.in_dim, IDENTITY)
        return out


class Prepared:
    """A dense layer whose weights are pre-quantized into the library's storage
    dtype and held resident narrow (half the bytes of float32 at bf16/fp16).
    Construct via Mantissa.prepare(). forward() narrows only the small input
    vector per call and runs the narrow C kernel (tk_linear_forward), so it
    skips the per-weight re-quantization that linear_forward does on every call.

    Results are bit-identical to tk_linear_forward. They differ from
    Mantissa.linear_forward by at most a ULP or two: both quantize through the
    same storage dtype, but the narrow kernel and the f32 path reduce in
    different orders (the reduction-order envelope documented in DESIGN.md)."""

    def __init__(self, lib, W, out_dim: int, in_dim: int, bias=None):
        self._lib = lib
        self.out_dim, self.in_dim = out_dim, in_dim
        ssz = lib.tk_scalar_size()
        self._W = self._narrow(W, out_dim * in_dim, ssz, "W")
        self._bias = self._narrow(bias, out_dim, ssz, "bias") if bias is not None else None
        self._xbuf = (ctypes.c_char * (ssz * in_dim))()   # reused per-call narrowed input

    def _narrow(self, src, n: int, ssz: int, name: str):
        srcc, _ = _as_c_float(src, n, name)
        dst = (ctypes.c_char * (ssz * n))()
        self._lib.tk_quantize(srcc, dst, n)
        return dst

    def forward(self, x, act: int, out=None):
        """y = act(W @ x + bias) using the resident narrow weights. x is narrowed
        into a reused scratch buffer (cheap: in_dim elements). Pass `out=` (a
        float32 numpy array or array('f') of out_dim) to avoid the per-call
        output allocation; otherwise a fresh list is returned."""
        xc, _ = _as_c_float(x, self.in_dim, "x")
        self._lib.tk_quantize(xc, self._xbuf, self.in_dim)
        if out is not None:
            yc, y_wb = _as_c_float(out, self.out_dim, "out")
            self._lib.tk_linear_forward(self._W, self._xbuf, self._bias, yc,
                                        self.out_dim, self.in_dim, act)
            if y_wb:
                y_wb()
            return out
        yc = (ctypes.c_float * self.out_dim)()
        self._lib.tk_linear_forward(self._W, self._xbuf, self._bias, yc,
                                    self.out_dim, self.in_dim, act)
        return list(yc)
