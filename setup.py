"""Build hook: compile src/*.c into libmantissa.<ext> and bundle it in the wheel.

Project metadata lives in pyproject.toml (PEP 621); this file only carries the
imperative bits setuptools cannot express declaratively — compiling the C core
with the same flags as the Makefile and shipping the result as package data.

The compiled library is not a CPython extension module (it is a plain C-ABI
shared library loaded via ctypes), so we drive the compiler directly rather than
through the extension machinery. A placeholder Extension is still declared so the
wheel is tagged platform-specific (not py3-none-any) and the build_ext command
runs. DTYPE defaults to 2 (bfloat16); override with MANTISSA_DTYPE=<id>.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:                                        # setuptools >=70.1 vendors it
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                         # older toolchains
    from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent.resolve()

# Mirrors SRC in the Makefile.
SOURCES = [
    "src/dtypes.c",
    "src/activations.c",
    "src/ops.c",
    "src/loss.c",
    "src/backprop.c",
    "src/pool.c",
    "src/conv.c",
]

DTYPE = os.environ.get("MANTISSA_DTYPE", "2")   # 2 = bfloat16 (Makefile default)


def _lib_filename() -> str:
    if sys.platform == "darwin":
        return "libmantissa.dylib"
    if sys.platform == "win32":
        return "libmantissa.dll"
    return "libmantissa.so"


class build_mantissa(build_ext):
    """Compile the C core into a ctypes shared library, same flags as the Makefile."""

    def build_extension(self, ext):
        return  # no CPython extension to build; the shared lib is built in run()

    def run(self):
        cc = os.environ.get("CC", "cc")
        libname = _lib_filename()

        # -O3 + fast FP contraction so tk_dot folds multiply-add into FMA and
        # vectorizes; -fvisibility=hidden + -DTK_BUILD_DLL export only the C ABI.
        cflags = [
            "-O3", "-funroll-loops", "-ffp-contract=fast",
            "-Wall", "-Wextra", "-std=c11",
            "-Iinclude", f"-DTK_DTYPE={DTYPE}", "-DTK_BUILD_DLL",
            "-fvisibility=hidden", "-pthread", "-fPIC", "-shared",
        ]
        ldflags = ["-lm", "-pthread"]
        srcs = [str(ROOT / s) for s in SOURCES]

        build_dir = Path(self.build_lib) / "mantissa"
        build_dir.mkdir(parents=True, exist_ok=True)
        out = build_dir / libname

        cmd = [cc, *cflags, "-o", str(out), *srcs, *ldflags]
        self.announce("building libmantissa: " + " ".join(cmd), level=2)
        subprocess.check_call(cmd)

        # For editable/develop installs (`pip install -e .`), the package is
        # imported from the source tree, so drop a copy there too.
        if self.inplace:
            dest = ROOT / "python" / "mantissa" / libname
            shutil.copy2(out, dest)

    def get_outputs(self):
        return [str(Path(self.build_lib) / "mantissa" / _lib_filename())]


class bdist_wheel_platlib(bdist_wheel):
    """Tag the wheel py3-none-<platform>: platform-specific (it ships a compiled
    library) but Python-version-agnostic (ctypes has no CPython ABI dependency),
    so one wheel per OS/arch serves every Python 3.x."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(
    # A sources-less placeholder so the build_ext command runs; the real
    # compilation happens in build_mantissa.run().
    ext_modules=[Extension("mantissa._libmantissa", sources=[])],
    cmdclass={"build_ext": build_mantissa, "bdist_wheel": bdist_wheel_platlib},
)
