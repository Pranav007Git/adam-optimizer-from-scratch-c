"""Python wrapper around the C neural network in src/nn.c (uses ctypes, builds the library with gcc if needed)."""

import ctypes
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "nn.c")

OPTIMIZERS = {"sgd": 0, "momentum": 1, "adam": 2}


def _lib_path():
    if os.name == "nt":
        ext = ".dll"
    elif sys.platform == "darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    return os.path.join(HERE, "libnn" + ext)


def build(force=False):
    """Compile src/nn.c into a shared library (skipped if it is already up to date)."""
    out = _lib_path()
    if not force and os.path.exists(out) and os.path.getmtime(out) >= max(
        os.path.getmtime(SRC), os.path.getmtime(os.path.join(HERE, "src", "nn.h"))
    ):
        return out
    cmd = ["gcc", "-O2", "-shared", "-o", out, SRC, "-lm"]
    if os.name != "nt":
        cmd.insert(3, "-fPIC")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "gcc was not found. Install a C compiler (Linux: sudo apt install build-essential; "
            "macOS: xcode-select --install; Windows: use WSL or MinGW-w64)."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Compiling src/nn.c failed:\n" + e.stderr)
    return out


_dbl = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

_lib = ctypes.CDLL(build())
_lib.net_create.restype = ctypes.c_void_p
_lib.net_create.argtypes = [_i32, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
_lib.net_free.restype = None
_lib.net_free.argtypes = [ctypes.c_void_p]
_lib.net_num_params.restype = ctypes.c_int
_lib.net_num_params.argtypes = [ctypes.c_void_p]
_lib.net_train_batch.restype = ctypes.c_double
_lib.net_train_batch.argtypes = [
    ctypes.c_void_p, _dbl, _i32, ctypes.c_int,
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.net_evaluate.restype = ctypes.c_double
_lib.net_evaluate.argtypes = [ctypes.c_void_p, _dbl, _i32, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
_lib.net_predict.restype = None
_lib.net_predict.argtypes = [ctypes.c_void_p, _dbl, ctypes.c_int, _i32]
_lib.net_grad_check.restype = ctypes.c_double
_lib.net_grad_check.argtypes = [ctypes.c_void_p, _dbl, _i32, ctypes.c_int, ctypes.c_double]


class Net:
    """Fully connected network (ReLU hidden layers, softmax output) that runs in C.

    Use one Net per optimizer: momentum and Adam keep their state inside the network.
    """

    def __init__(self, sizes, max_batch=512, seed=0):
        self.sizes = [int(s) for s in sizes]
        arr = np.asarray(self.sizes, dtype=np.int32)
        self._h = _lib.net_create(arr, len(arr), int(max_batch), int(seed) & 0xFFFFFFFF)
        if not self._h:
            raise ValueError("net_create failed: check the layer sizes (2-8 layers, all > 0) and memory")
        self.max_batch = int(max_batch)

    def close(self):
        if getattr(self, "_h", None):
            _lib.net_free(self._h)
            self._h = None

    def __del__(self):
        self.close()

    @property
    def num_params(self):
        return _lib.net_num_params(self._h)

    def _prep(self, X, y=None):
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.sizes[0]:
            raise ValueError(f"X must have shape (n, {self.sizes[0]}), got {X.shape}")
        if y is None:
            return X
        y = np.ascontiguousarray(y, dtype=np.int32)
        if y.shape != (X.shape[0],):
            raise ValueError("y must be a 1-D array with one label per row of X")
        if y.min() < 0 or y.max() >= self.sizes[-1]:
            raise ValueError("labels must be in 0 .. n_classes-1")
        return X, y

    def train_batch(self, X, y, optimizer="adam", lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """One update step on one mini-batch. Returns the batch loss before the update."""
        X, y = self._prep(X, y)
        if len(X) > self.max_batch:
            raise ValueError("batch is larger than max_batch")
        loss = _lib.net_train_batch(self._h, X, y, len(X), OPTIMIZERS[optimizer], lr, beta1, beta2, eps)
        if loss == -1.0:
            raise ValueError("net_train_batch rejected its arguments")
        return loss

    def evaluate(self, X, y):
        """Returns (mean loss, accuracy)."""
        X, y = self._prep(X, y)
        acc = ctypes.c_double()
        loss = _lib.net_evaluate(self._h, X, y, len(X), ctypes.byref(acc))
        return loss, acc.value

    def predict(self, X):
        X = self._prep(X)
        out = np.zeros(len(X), dtype=np.int32)
        _lib.net_predict(self._h, X, len(X), out)
        return out

    def grad_check(self, X, y, h=1e-5):
        """Worst relative error between backprop and numerical gradients."""
        X, y = self._prep(X, y)
        return _lib.net_grad_check(self._h, X, y, len(X), h)
