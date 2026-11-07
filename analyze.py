"""Turns results/runs.csv and results/curves.csv into plots and a summary table.

Outputs (in the results folder):
  fig1_lr_sensitivity.png  test accuracy vs learning rate, one panel per batch size
  fig2_convergence.png     training loss and validation accuracy of each optimizer's best setting
  summary.md               tables used in the write-up
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"sgd": "tab:blue", "momentum": "tab:green", "adam": "tab:red"}
LABELS = {"sgd": "SGD", "momentum": "SGD + Momentum", "adam": "Adam"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--target-acc", type=float, default=0.97, help="validation accuracy used for 'epochs to reach'")
    p.add_argument("--ok-acc", type=float, default=0.90, help="mean test accuracy that counts as 'works' for a learning rate")
    args = p.parse_args()

    runs = pd.read_csv(os.path.join(args.results, "runs.csv"))
    curves = pd.read_csv(os.path.join(args.results, "curves.csv"))
    opts = [o for o in ["sgd", "momentum", "adam"] if o in set(runs.optimizer)]
    batches = sorted(runs.batch.unique())

    # ---- figure 1: learning-rate sensitivity ----
    fig, axes = plt.subplots(1, len(batches), figsize=(4.6 * len(batches), 3.8), sharey=True, squeeze=False)
    for ax, bs in zip(axes[0], batches):
        for o in opts:
            g = runs[(runs.optimizer == o) & (runs.batch == bs)].groupby("lr").final_test_acc.agg(["mean", "std"]).fillna(0)
            ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color=COLORS[o], label=LABELS[o])
        ax.set_xscale("log")
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("learning rate")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test accuracy (mean ± std over seeds)")
    axes[0][0].legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig1_lr_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ---- best setting per optimizer, chosen on VALIDATION accuracy ----
    cfg = runs.groupby(["optimizer", "lr", "batch"]).agg(
        val=("final_val_acc", "mean"), test=("final_test_acc", "mean"), test_std=("final_test_acc", "std"),
        diverged=("diverged", "sum"), n=("seed", "count")).reset_index()
    best = {o: cfg[cfg.optimizer == o].sort_values("val", ascending=False).iloc[0] for o in opts}

    # ---- figure 2: convergence of the best settings ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    epochs_to_target = {}
    for o in opts:
        b = best[o]
        c = curves[(curves.optimizer == o) & (curves.lr == b.lr) & (curves.batch == b.batch)]
        m = c.groupby("epoch")[["train_loss", "val_acc"]].mean()
        a1.plot(m.index, m.train_loss, color=COLORS[o], label=f"{LABELS[o]} (lr={b.lr:g}, batch={int(b.batch)})")
        a2.plot(m.index, m.val_acc, color=COLORS[o])
        hit = m.index[m.val_acc >= args.target_acc]
        epochs_to_target[o] = int(hit[0]) if len(hit) else None
    a1.set_yscale("log")
    a1.set_xlabel("epoch"); a1.set_ylabel("training loss"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.set_xlabel("epoch"); a2.set_ylabel("validation accuracy"); a2.grid(alpha=0.3)
    a2.axhline(args.target_acc, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "fig2_convergence.png"), dpi=150)
    plt.close(fig)

    # ---- summary tables ----
    lines = ["# Results summary", ""]
    lines += ["## Best setting per optimizer (chosen on validation accuracy, reported on test accuracy)", "",
              "| Optimizer | Learning rate | Batch size | Test acc (mean ± std over seeds) | Epochs to reach "
              f"{args.target_acc:.0%} val acc |", "|---|---|---|---|---|"]
    for o in opts:
        b = best[o]
        e = epochs_to_target[o]
        lines.append(f"| {LABELS[o]} | {b.lr:g} | {int(b.batch)} | {b.test:.4f} ± {0 if pd.isna(b.test_std) else b.test_std:.4f} | "
                     f"{e if e is not None else 'not reached'} |")

    lines += ["", f"## How many learning rates work? (mean test accuracy over batch sizes and seeds ≥ {args.ok_acc:.0%})", "",
              "| Optimizer | Learning rates that work | Count |", "|---|---|---|"]
    for o in opts:
        g = runs[runs.optimizer == o].groupby("lr").final_test_acc.mean()
        good = [f"{lr:g}" for lr, v in g.items() if v >= args.ok_acc]
        lines.append(f"| {LABELS[o]} | {', '.join(good) if good else 'none'} | {len(good)} of {len(g)} |")

    lines += ["", "## Diverged runs (loss became NaN or infinity)", "", "| Optimizer | Diverged runs | Total runs |", "|---|---|---|"]
    for o in opts:
        r = runs[runs.optimizer == o]
        lines.append(f"| {LABELS[o]} | {int(r.diverged.sum())} | {len(r)} |")

    lines += ["", "## Mean test accuracy by learning rate (averaged over batch sizes and seeds)", "",
              "| Learning rate | " + " | ".join(LABELS[o] for o in opts) + " |", "|---|" + "---|" * len(opts)]
    pivot = runs.groupby(["lr", "optimizer"]).final_test_acc.mean().unstack()
    for lr, row in pivot.iterrows():
        lines.append(f"| {lr:g} | " + " | ".join(f"{row[o]:.4f}" for o in opts) + " |")

    with open(os.path.join(args.results, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.results}/fig1_lr_sensitivity.png, fig2_convergence.png, summary.md")


if __name__ == "__main__":
    main()
