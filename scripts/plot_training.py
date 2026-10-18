"""Plot recorded training and fixed-validation losses without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def plot(run_name: str, window: int = 100):
    run = ROOT / "05_Models/training_runs" / run_name
    log = run / "train_log.jsonl"
    validation = run / "validation_history.json"
    config = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    history = json.loads(validation.read_text())
    if not rows:
        raise ValueError("No recorded optimizer updates to plot")
    steps = np.array([row["step"] for row in rows])
    losses = np.array([row["loss"] for row in rows])
    if not np.isfinite(losses).all() or (np.diff(steps) <= 0).any():
        raise ValueError("Losses must be finite and recorded steps strictly increasing")
    window = max(1, min(int(window), len(rows)))
    mean = np.convolve(losses, np.ones(window) / window, mode="valid")
    completed = (run / "summary.json").exists()
    summary = json.loads((run / "summary.json").read_text()) if completed else None
    status = "completed" if completed else "in progress"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    axes[0].plot(
        steps, losses, color="#64748b", alpha=0.3, linewidth=0.7, label="Recorded batch MSE"
    )
    axes[0].plot(
        steps[window - 1 :],
        mean,
        color="#0369a1",
        linewidth=2,
        label=f"{window}-update moving mean",
    )
    axes[0].set(
        xlabel="Optimizer update",
        ylabel="Training noise-prediction MSE",
        title="Stochastic training loss",
    )
    axes[0].legend(frameon=False)
    val_steps = [row["step"] for row in history]
    val_scores = [row["validation_mse"] for row in history]
    axes[1].plot(val_steps, val_scores, marker="o", color="#0f766e", linewidth=2)
    axes[1].set(
        xlabel="Optimizer update (0 = pretrained base)",
        ylabel="Fixed validation MSE",
        title="Same 64 validation photos and noise draws",
    )
    if summary:
        axes[1].scatter(
            [summary["best_step"]],
            [summary["best_validation_mse"]],
            s=100,
            facecolors="none",
            edgecolors="#dc2626",
            linewidths=2,
            label=f"Selected: step {summary['best_step']}",
        )
        axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle(
        f"CatStudio LoRA | {run_name} | {steps[-1]:,}/{config['max_steps']:,} updates | {status}",
        fontsize=14,
    )
    out = ROOT / "06_Results/training_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{run_name}_loss_curves.png", dpi=180)
    fig.savefig(out / f"{run_name}_loss_curves.pdf")
    plt.close(fig)
    evidence = {
        "run": run_name,
        "status": status,
        "recorded_updates": len(rows),
        "last_step": int(steps[-1]),
        "moving_mean_window": window,
        "training_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "validation_history_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "validation_history": history,
        "note": "Training uses stochastic photo/timestep/noise batches; validation uses fixed photos and noise. The two panels are separate measurements. Lower denoising MSE does not establish better image quality.",
    }
    (out / f"{run_name}_loss_curves.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "file": str(out / f"{run_name}_loss_curves.png"),
                "status": status,
                "last_step": int(steps[-1]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    plot(args.run_name, args.window)
