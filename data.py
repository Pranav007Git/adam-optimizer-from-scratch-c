"""Dataset loading: 'digits' (built into scikit-learn, works offline) or 'mnist' (downloaded once)."""

import numpy as np
from sklearn.datasets import fetch_openml, load_digits


def load_dataset(name="digits", n_total=None, seed=0):
    """Returns a dict with train / validation / test splits (60% / 20% / 20%), standardised using train statistics.

    The split is the same for every experiment seed, so seeds only change weight
    initialisation and batch order.
    """
    if name == "digits":
        d = load_digits()
        X, y = d.data / 16.0, d.target
    elif name == "mnist":
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        X, y = X / 255.0, y.astype(int)
    else:
        raise ValueError("dataset must be 'digits' or 'mnist'")

    idx = np.random.RandomState(seed).permutation(len(X))
    if n_total:
        idx = idx[:n_total]
    X, y = X[idx], y[idx]

    n = len(X)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    parts = {
        "train": (X[:n_tr], y[:n_tr]),
        "val": (X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]),
        "test": (X[n_tr + n_va:], y[n_tr + n_va:]),
    }

    mean = parts["train"][0].mean(axis=0)
    std = parts["train"][0].std(axis=0)
    std[std < 1e-8] = 1.0  # constant pixels

    out = {"n_features": X.shape[1], "n_classes": int(y.max()) + 1}
    for k, (Xs, ys) in parts.items():
        out["X_" + k] = np.ascontiguousarray((Xs - mean) / std, dtype=np.float64)
        out["y_" + k] = np.ascontiguousarray(ys, dtype=np.int32)
    return out
