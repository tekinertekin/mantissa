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
from pathlib import Path

# Activation ids must match include/activations.h.
IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU = range(7)


def _library_path() -> str:
    ext = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
    root = Path(__file__).resolve().parent.parent
    # Prefer the committed prebuilt library; fall back to a local `make dist`.
    for lib in (root / "dist" / f"libmantissa.{ext}", root / "build" / f"libmantissa.{ext}"):
        if lib.exists():
            return str(lib)
    raise FileNotFoundError(
        f"libmantissa.{ext} not found in dist/ or build/. Build it:  make dist"
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

    @property
    def dtype(self) -> str:
        """Storage type the loaded library was compiled for (e.g. 'bfloat16')."""
        return self._lib.tk_dtype_name().decode()

    def linear_forward(self, W, x, bias, out_dim: int, in_dim: int, act: int):
        """y = act(W @ x + bias). W is row-major out_dim x in_dim; bias may be None.

        Values are quantized through the library's storage dtype internally."""
        Wc = (ctypes.c_float * (out_dim * in_dim))(*W)
        xc = (ctypes.c_float * in_dim)(*x)
        yc = (ctypes.c_float * out_dim)()
        bc = (ctypes.c_float * out_dim)(*bias) if bias is not None else None
        self._lib.tk_linear_forward_f32(Wc, xc, bc, yc, out_dim, in_dim, act)
        return list(yc)

    def train_step(self, W, x, target, out_dim: int, in_dim: int,
                   act: int, lr: float, bias=None) -> float:
        """One SGD step on a dense layer. Mutates W (and bias) in place, returns
        the MSE loss before the update. W is row-major out_dim x in_dim."""
        Wc = (ctypes.c_float * (out_dim * in_dim))(*W)
        xc = (ctypes.c_float * in_dim)(*x)
        tc = (ctypes.c_float * out_dim)(*target)
        bc = (ctypes.c_float * out_dim)(*bias) if bias is not None else None
        loss = self._lib.tk_train_step_f32(Wc, bc, xc, tc, out_dim, in_dim, act, lr)
        W[:] = list(Wc)                       # reflect in-place update back to caller
        if bias is not None:
            bias[:] = list(bc)
        return float(loss)
