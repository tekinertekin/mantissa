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

__all__ = ["Mantissa", "IDENTITY", "STEP", "SIGN", "RELU", "SIGMOID", "TANH", "GELU"]

# Activation ids must match include/activations.h.
IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU = range(7)

_f32p = ctypes.POINTER(ctypes.c_float)


def _as_c_float(seq, n: int):
    """Expose `seq` (length n) to C as float32 without a per-element Python copy
    when possible. Returns (arg, writeback): `arg` is what to pass to the C call;
    if `writeback` is not None, call it after the C call to reflect in-place
    mutation back to `seq`.

    Zero-copy (mutation is seen by the caller automatically, writeback=None) for
    a C-contiguous float32 numpy.ndarray or an array('f'); a plain list/tuple is
    boxed into a ctypes buffer and, if the C call mutates it, copied back."""
    if _np is not None and isinstance(seq, _np.ndarray) \
            and seq.dtype == _np.float32 and seq.flags["C_CONTIGUOUS"]:
        return seq.ctypes.data_as(_f32p), None
    if isinstance(seq, array) and seq.typecode == "f":
        return (ctypes.c_float * n).from_buffer(seq), None
    buf = (ctypes.c_float * n)(*seq)                # list/tuple/other: one copy
    return buf, (lambda: seq.__setitem__(slice(None), buf))


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
        f32p = ctypes.POINTER(ctypes.c_float)

        self._lib.tk_dtype_name.restype = ctypes.c_char_p
        self._lib.tk_scalar_size.restype = ctypes.c_int

        self._lib.tk_linear_forward_f32.restype = None
        self._lib.tk_linear_forward_f32.argtypes = [
            f32p, f32p, f32p, f32p,           # W, x, bias, y
            ctypes.c_int, ctypes.c_int,       # out_dim, in_dim
            ctypes.c_int,                     # activation
        ]

        # one float32 SGD training step (forward + backward + update); returns loss
        self._lib.tk_train_step_f32.restype = ctypes.c_float
        self._lib.tk_train_step_f32.argtypes = [
            f32p, f32p, f32p, f32p,           # W, bias, x, target
            ctypes.c_int, ctypes.c_int,       # out_dim, in_dim
            ctypes.c_int, ctypes.c_float,     # activation, lr
        ]

        # a whole epoch of sequential SGD in one call; returns the mean loss
        self._lib.tk_train_epoch_f32.restype = ctypes.c_float
        self._lib.tk_train_epoch_f32.argtypes = [
            f32p, f32p, f32p, f32p,           # W, bias, X, targets
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_samples, out_dim, in_dim
            ctypes.c_int, ctypes.c_float,     # activation, lr
        ]

    @property
    def dtype(self) -> str:
        """Storage type the loaded library was compiled for (e.g. 'bfloat16')."""
        return self._lib.tk_dtype_name().decode()

    def linear_forward(self, W, x, bias, out_dim: int, in_dim: int, act: int):
        """y = act(W @ x + bias). W is row-major out_dim x in_dim; bias may be None.

        Values are quantized through the library's storage dtype internally.
        Pass W/x/bias as float32 numpy arrays or array('f') to skip the per-
        element copy (see _as_c_float)."""
        Wc, _ = _as_c_float(W, out_dim * in_dim)      # read-only inputs: no writeback
        xc, _ = _as_c_float(x, in_dim)
        yc = (ctypes.c_float * out_dim)()
        bc = _as_c_float(bias, out_dim)[0] if bias is not None else None
        self._lib.tk_linear_forward_f32(Wc, xc, bc, yc, out_dim, in_dim, act)
        return list(yc)

    def train_step(self, W, x, target, out_dim: int, in_dim: int,
                   act: int, lr: float, bias=None) -> float:
        """One SGD step on a dense layer. Mutates W (and bias) in place, returns
        the MSE loss before the update. W is row-major out_dim x in_dim.

        Pass W/bias as float32 numpy arrays or array('f') for a zero-copy update
        (mutated directly, no list round-trip); a plain list is copied in and
        copied back."""
        Wc, W_wb = _as_c_float(W, out_dim * in_dim)   # mutated -> may need writeback
        xc, _ = _as_c_float(x, in_dim)
        tc, _ = _as_c_float(target, out_dim)
        bc, b_wb = _as_c_float(bias, out_dim) if bias is not None else (None, None)
        loss = self._lib.tk_train_step_f32(Wc, bc, xc, tc, out_dim, in_dim, act, lr)
        if W_wb:                              # reflect update back for the copy path
            W_wb()
        if b_wb:
            b_wb()
        return float(loss)

    def train_epoch(self, W, X, targets, n_samples: int, out_dim: int,
                    in_dim: int, act: int, lr: float, bias=None) -> float:
        """A full epoch of sequential SGD in ONE C call: X is n_samples rows of
        in_dim, targets n_samples rows of out_dim (both flat, row-major).
        Weight updates are numerically identical to calling train_step once per
        sample; the win is one FFI crossing per epoch instead of one per sample.
        Mutates W (and bias) in place; returns the mean pre-update loss."""
        Wc, W_wb = _as_c_float(W, out_dim * in_dim)
        Xc, _ = _as_c_float(X, n_samples * in_dim)
        tc, _ = _as_c_float(targets, n_samples * out_dim)
        bc, b_wb = _as_c_float(bias, out_dim) if bias is not None else (None, None)
        loss = self._lib.tk_train_epoch_f32(Wc, bc, Xc, tc,
                                            n_samples, out_dim, in_dim, act, lr)
        if W_wb:
            W_wb()
        if b_wb:
            b_wb()
        return float(loss)
