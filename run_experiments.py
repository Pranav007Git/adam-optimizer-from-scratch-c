"""Runs the optimizer experiments: SGD vs Momentum vs Adam over learning rates, batch sizes and seeds.

Writes results/runs.csv (one row per run) and results/curves.csv (one row per epoch per run).
Then run:  python analyze.py
"""

import argparse
import csv
import itertools
import math
import os
import time

import numpy as np

from data import load_dataset
from nn import Net


def run_one(data, optimizer, lr, batch, seed, epochs, hidden):
    net = Net([data["n_features"], *hidden, data["n_classes"]], max_batch=512, seed=seed)
    rng = np.random.RandomState(seed)
    Xtr, ytr = data["X_train"], data["y_train"]
    n = len(Xtr)

    curve = []
    diverged = False
    for epoch in range(1, epochs + 1):
        train_loss = float("nan")
        if not diverged:
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                loss = net.train_batch(Xtr[b], ytr[b], optimizer, lr)
                if not math.isfinite(loss):
                    diverged = True
                    break
                total += loss * len(b)
            if not diverged:
                train_loss = total / n
        _, val_acc = net.evaluate(data["X_val"], data["y_val"])
        curve.append((epoch, train_loss, val_acc))

    _, test_acc = net.evaluate(data["X_test"], data["y_test"])
    return curve, diverged, test_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    p.add_argument("--n-total", type=int, default=None, help="use only this many samples (useful for mnist, e.g. 10000)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, nargs="+", default=[64], help="hidden layer sizes")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--optimizers", nargs="+", default=["sgd", "momentum", "adam"])
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_dataset(args.dataset, args.n_total)
    print(f"dataset={args.dataset}  train={len(data['X_train'])}  val={len(data['X_val'])}  test={len(data['X_test'])}")

    grid = list(itertools.product(args.optimizers, args.lrs, args.batch_sizes, args.seeds))
    print(f"{len(grid)} runs x {args.epochs} epochs, hidden layers = {args.hidden}")

    runs_f = open(os.path.join(args.out, "runs.csv"), "w", newline="")
    curves_f = open(os.path.join(args.out, "curves.csv"), "w", newline="")
    runs_w, curves_w = csv.writer(runs_f), csv.writer(curves_f)
    runs_w.writerow(["optimizer", "lr", "batch", "seed", "diverged", "final_train_loss",
                     "final_val_acc", "final_test_acc", "seconds"])
    curves_w.writerow(["optimizer", "lr", "batch", "seed", "epoch", "train_loss", "val_acc"])

    start = time.perf_counter()
    for i, (opt, lr, bs, seed) in enumerate(grid, 1):
        t0 = time.perf_counter()
        curve, diverged, test_acc = run_one(data, opt, lr, bs, seed, args.epochs, args.hidden)
        secs = time.perf_counter() - t0
        runs_w.writerow([opt, lr, bs, seed, int(diverged), curve[-1][1], curve[-1][2], test_acc, round(secs, 3)])
        for epoch, tl, va in curve:
            curves_w.writerow([opt, lr, bs, seed, epoch, tl, va])
        print(f"[{i:3d}/{len(grid)}] {opt:8s} lr={lr:<7g} batch={bs:<4d} seed={seed}  "
              f"test_acc={test_acc:.4f}{'  DIVERGED' if diverged else ''}  ({secs:.1f}s)")
        runs_f.flush()
        curves_f.flush()

    runs_f.close()
    curves_f.close()
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}/runs.csv, {args.out}/curves.csv")


if __name__ == "__main__":
    main()
