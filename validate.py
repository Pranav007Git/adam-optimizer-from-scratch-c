"""Checks that the C network is correct.

1. Gradient check: backprop gradients must match numerical gradients.
2. Reference check: same architecture and optimizer in scikit-learn should reach a similar test accuracy.
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from data import load_dataset
from nn import Net


def gradient_check():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 6)
    y = rng.randint(0, 4, size=12)
    net = Net([6, 9, 7, 4], max_batch=12, seed=1)
    err = net.grad_check(X, y, h=1e-5)
    ok = err < 1e-6
    print(f"[1] gradient check   worst relative error = {err:.2e}   {'PASS' if ok else 'FAIL'} (needs < 1e-6)")
    return ok


def reference_check(epochs=30, batch=64, lr=1e-3, hidden=64, seed=0):
    d = load_dataset("digits")
    rng = np.random.RandomState(seed)

    ours = Net([d["n_features"], hidden, d["n_classes"]], max_batch=512, seed=seed)
    n = len(d["X_train"])
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            b = perm[s:s + batch]
            ours.train_batch(d["X_train"][b], d["y_train"][b], "adam", lr)
    _, our_acc = ours.evaluate(d["X_test"], d["y_test"])

    ref = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", solver="adam", alpha=0.0,
                        batch_size=batch, learning_rate_init=lr, max_iter=epochs, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ref.fit(d["X_train"], d["y_train"])
    ref_acc = ref.score(d["X_test"], d["y_test"])

    gap = abs(our_acc - ref_acc)
    ok = gap < 0.03
    print(f"[2] reference check  ours = {our_acc:.4f}   scikit-learn = {ref_acc:.4f}   "
          f"gap = {gap:.4f}   {'PASS' if ok else 'FAIL'} (needs gap < 0.03)")
    return ok


if __name__ == "__main__":
    results = [gradient_check(), reference_check()]
    raise SystemExit(0 if all(results) else 1)
