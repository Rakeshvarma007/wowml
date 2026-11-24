"""
_bindings.py — Load the compiled C shared library via ctypes.
"""
import ctypes
import sys
import pathlib


def _find_library() -> pathlib.Path:
    """Locate the compiled wowml_c shared library."""
    here = pathlib.Path(__file__).resolve().parent

    if sys.platform == "win32":
        names = ["wowml_c.dll", "wowml_c.pyd"]
    elif sys.platform == "darwin":
        names = ["wowml_c.dylib", "libwowml_c.dylib"]
    else:
        names = ["wowml_c.so", "libwowml_c.so"]

    search_dirs = [
        here,
        here.parent.parent / "build",
        here.parent.parent / "build" / "Release",
    ]

    for d in search_dirs:
        for name in names:
            path = d / name
            if path.exists():
                return path

    raise OSError(
        "Could not find wowml_c shared library. "
        "Build first: mkdir build && cd build && cmake .. && make"
    )


def _load_library():
    path = _find_library()
    return ctypes.CDLL(str(path))


try:
    lib = _load_library()
except OSError:
    lib = None  # Library not yet built

ffi = None  # Placeholder for future CFFI
