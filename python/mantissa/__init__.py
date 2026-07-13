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
from array import array
from pathlib import Path

try:                                   # optional: enables a zero-copy fast path
    import numpy as _np
except ImportError:                    # binding works without it (plain lists)
    _np = None

__all__ = ["Mantissa", "Prepared", "Trainer",
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

        # one epoch of the mistake-driven Rosenblatt rule; returns mistakes
        self._lib.tk_perceptron_epoch_f32.restype = ctypes.c_int
        self._lib.tk_perceptron_epoch_f32.argtypes = [
            _f32p, _f32p, _f32p, _f32p,           # W, bias, X, targets
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_samples, out_dim, in_dim
            ctypes.c_float, _i32p,            # lr, order (or NULL)
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
