"""Thin ctypes binding over the mantissa shared library.

The library is built for one storage dtype at a time (see config.h). This
wrapper assumes it was built as float32 (`make lib`, the default), so the
Python side passes plain 32-bit float buffers. Build a bf16/fp8 variant and the
same C code runs; only the buffer packing on the Python side would change.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Activation ids must match include/activations.h.
IDENTITY, STEP, SIGN, RELU, SIGMOID, TANH, GELU = range(7)


def _library_path() -> str:
    ext = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
    build = Path(__file__).resolve().parent.parent / "build" / f"libmantissa.{ext}"
    if not build.exists():
        raise FileNotFoundError(
            f"{build} not found. Build it first:  make lib"
        )
    return str(build)


class Mantissa:
    def __init__(self, path: str | None = None):
        self._lib = ctypes.CDLL(path or _library_path())
        f32p = ctypes.POINTER(ctypes.c_float)

        self._lib.tk_dtype_name.restype = ctypes.c_char_p
        self._lib.tk_scalar_size.restype = ctypes.c_int

        self._lib.tk_linear_forward.restype = None
        self._lib.tk_linear_forward.argtypes = [
            f32p, f32p, f32p, f32p,           # W, x, bias, y
            ctypes.c_int, ctypes.c_int,       # out_dim, in_dim
            ctypes.c_int,                     # activation
        ]

        if self.dtype != "float32":
            raise RuntimeError(
                f"binding expects a float32 build, got '{self.dtype}'. "
                f"Rebuild with `make lib` (DTYPE=0)."
            )

    @property
    def dtype(self) -> str:
        return self._lib.tk_dtype_name().decode()

    def linear_forward(self, W, x, bias, out_dim: int, in_dim: int, act: int):
        """y = act(W @ x + bias). W is row-major out_dim x in_dim; bias may be None."""
        Wc = (ctypes.c_float * (out_dim * in_dim))(*W)
        xc = (ctypes.c_float * in_dim)(*x)
        yc = (ctypes.c_float * out_dim)()
        bc = (ctypes.c_float * out_dim)(*bias) if bias is not None else None
        self._lib.tk_linear_forward(Wc, xc, bc, yc, out_dim, in_dim, act)
        return list(yc)
