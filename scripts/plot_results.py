"""Render report plots from completed run/evaluation evidence, never synthetic results."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_Results/figures"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(training, summary, plan, history, metadata):
    """Bind plotted values and images to the completed, selected experiment."""
    plan_hash = digest(ROOT / "03_Code/config/evaluation_plan.json")
    if plan_hash != summary["evaluation_plan_sha256"]:
        raise ValueError("Evaluation-plan identity mismatch")
    for filename, field in (
        ("per_image_metrics.json", "per_image_metrics_sha256"),
        ("generated_clip_features.npy", "generated_clip_features_sha256"),
    ):
        if digest(ROOT / "06_Results/evaluation" / filename) != summary[field]:
            raise ValueError(f"Stale evaluation summary: {filename} identity mismatch")
    adapter_hash = digest(ROOT / "05_Models/cat_lora/pytorch_lora_weights.safetensors")
    provenance = read(ROOT / "05_Models/cat_lora/ADAPTER_PROVENANCE.json")
    selected = ROOT / training["best_adapter_path"] / "pytorch_lora_weights.safetensors"
    base_revision = read(ROOT / "05_Models/base_sd15/PROVENANCE.json")["revision"]
    if (
        adapter_hash != summary["adapter_sha256"]
        or adapter_hash != provenance["adapter_sha256"]
        or adapter_hash != digest(selected)
        or provenance["training_summary"] != training
        or summary["base_revision"] != base_revision
        or training["base_revision"] != base_revision
    ):
        raise ValueError("Selected training checkpoint, adapter or base identity mismatch")
    trained = [r for r in history if r["step"] > 0]
    if (
        not trained
        or len({r["step"] for r in history}) != len(history)
        or any(not np.isfinite(r["validation_mse"]) for r in history)
        or max(r["step"] for r in history) != training["completed_steps"]
    ):
        raise ValueError("Incomplete or invalid validation history")
    best = min(trained, key=lambda r: r["validation_mse"])
    if best["step"] != training["best_step"] or not np.isclose(
        best["validation_mse"], training["best_validation_mse"], rtol=0, atol=1e-12
    ):
        raise ValueError("Selected checkpoint is not the minimum among trained checkpoints")
    expected = {
        (c, p["id"], s) for c in ("base", "lora") for p in plan["prompts"] for s in plan["seeds"]
    }
    lookup = {(r["model_choice"], r["prompt_id"], r["seed"]): r for r in metadata}
    if len(metadata) != 128 or len(lookup) != len(metadata) or set(lookup) != expected:
        raise ValueError("Missing, duplicate or unexpected image records in the paired grid")
    for choice, pid, seed in sorted(expected):
        row = lookup[(choice, pid, seed)]
        prompt = next(p for p in plan["prompts"] if p["id"] == pid)
        relative = f"06_Results/evaluation/images/{choice}/{pid}_s{seed}.png"
        wanted = {
            "file": relative,
            "prompt": prompt["prompt"],
            "base_revision": base_revision,
            "evaluation_plan_sha256": plan_hash,
            "num_steps": plan["num_steps"],
            "guidance_scale": plan["guidance_scale"],
            "width": plan["width"],
            "height": plan["height"],
            "adapter_scale": 1.0 if choice == "lora" else 0.0,
            "scheduler": "DPMSolverMultistepScheduler",
            "use_karras_sigmas": True,
            "negative_prompt": None,
            "safety_checker_enabled": True,
        }
        if choice == "lora":
            wanted["adapter_sha256"] = adapter_hash
        if (
            any(row.get(k) != v for k, v in wanted.items())
            or digest(ROOT / relative) != row["image_sha256"]
        ):
            raise ValueError(f"Stale or altered generated image: {choice}/{pid}/seed{seed}")
    for key, row in summary["metrics"].items():
        values = [
            row["base_mean"],
            row["lora_mean"],
            row["paired_delta_lora_minus_base"],
            *row["paired_prompt_cluster_bootstrap_95pct_ci"],
        ]
        if (
            len(values) != 5
            or not np.isfinite(values).all()
            or row["prompt_clusters"] != 16
            or row["bootstrap_replicates"] != 5000
            or values[3] > values[4]
            or not np.isclose(values[2], values[1] - values[0], rtol=0, atol=1e-10)
        ):
            raise ValueError(f"Invalid paired metric summary: {key}")
    return lookup


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    run = ROOT / "05_Models/training_runs/full"
    training = read(run / "summary.json")
    summary = read(ROOT / "06_Results/evaluation/summary.json")
    if (
        training["status"] != "TRAINING_COMPLETED"
        or training["completed_steps"] != training["max_steps"]
    ):
        raise ValueError("Figures require completed training")
    if (
        summary["image_pairs"] != 64
        or summary["prompt_clusters"] != 16
        or summary["seeds_per_prompt"] != 4
    ):
        raise ValueError("Figures require the complete frozen evaluation grid")
    plan = ROOT / "03_Code/config/evaluation_plan.json"
    history = read(run / "validation_history.json")
    metadata = read(ROOT / "06_Results/evaluation/per_image_metrics.json")
    lookup = verify_inputs(training, summary, read(plan), history, metadata)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    steps = [row["step"] for row in history]
    scores = [row["validation_mse"] for row in history]
    ax.plot(steps, scores, "o-", color="#235c75", label="Fixed validation MSE")
    ax.scatter(
        [training["best_step"]],
        [training["best_validation_mse"]],
        s=110,
        facecolor="none",
        edgecolor="#bd6529",
        linewidth=2,
        label="Selected trained checkpoint",
        zorder=5,
    )
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Noise-prediction MSE",
        title="Checkpoint selection on 64 validation images",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "validation_curve")

    definitions = [
        ("clip_text_cosine", "Image-text cosine"),
        ("color_proxy_agreement", "Color agreement proxy"),
        ("pattern_proxy_agreement", "Pattern agreement proxy"),
        ("pose_proxy_agreement", "Pose agreement proxy"),
        ("within_prompt_clip_cosine_distance_diversity", "Within-prompt diversity"),
    ]
    fig, axes = plt.subplots(1, len(definitions), figsize=(12.4, 3.3), layout="constrained")
    for ax, (key, label) in zip(axes, definitions):
        row = summary["metrics"][key]
        vals = [row["base_mean"], row["lora_mean"]]
        bars = ax.bar(["Base", "LoRA"], vals, color=["#8e9ca6", "#235c75"], width=0.62)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(
            (0, 1)
            if "agreement" in key
            else (min(0, min(vals) * 1.25), max(0.01, max(vals) * 1.25))
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        for b, value in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    fig.suptitle("Automated proxies on 16 prompts × 4 seeds", fontsize=12)
    save(fig, "metric_means")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), layout="constrained")
    groups = [definitions[:1] + definitions[4:], definitions[1:4]]
    for ax, group in zip(axes, groups):
        for i, (key, label) in enumerate(group):
            row = summary["metrics"][key]
            mean = row["paired_delta_lora_minus_base"]
            lo, hi = row["paired_prompt_cluster_bootstrap_95pct_ci"]
            ax.plot([lo, hi], [i, i], color="#235c75", lw=2)
            ax.plot([mean], [i], "o", color="#235c75")
        ax.set_yticks(range(len(group)), [label for _, label in group], fontsize=9)
        ax.axvline(0, color="#777777", lw=1, linestyle="--")
        ax.set_xlabel("LoRA minus base")
        ax.set_ylim(-0.6, len(group) - 0.4)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Paired differences: 95% prompt-cluster bootstrap intervals", fontsize=12)
    save(fig, "paired_differences")

    # Illustrative examples selected by prompt type, before seeing final outputs.
    # They do not replace the complete 128-image grid or the fixed visual audit.
    prompts = {p["id"]: p for p in read(plan)["prompts"]}
    for pid in ("p03", "p15"):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.15), layout="constrained")
        for ax, choice in zip(axes, ("base", "lora")):
            row = lookup[(choice, pid, 101)]
            source = ROOT / row["file"]
            ax.imshow(plt.imread(source))
            ax.axis("off")
            ax.set_title("Base" if choice == "base" else "CatStudio LoRA")
        fig.suptitle(
            f"{pid} · seed 101 · {prompts[pid]['color']} {prompts[pid]['pattern']} cat · {prompts[pid]['pose']} view",
            fontsize=11,
        )
        save(fig, f"paired_{pid}_seed101")
    sources = [
        run / "summary.json",
        run / "validation_history.json",
        ROOT / "06_Results/evaluation/summary.json",
        ROOT / "06_Results/evaluation/per_image_metrics.json",
        ROOT / "06_Results/evaluation/generated_clip_features.npy",
        ROOT / "05_Models/cat_lora/ADAPTER_PROVENANCE.json",
        plan,
    ]
    receipt = {
        "source_hashes": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "matplotlib_version": matplotlib.__version__,
        "plot_source_sha256": digest(Path(__file__)),
        "adapter_sha256": summary["adapter_sha256"],
        "base_revision": summary["base_revision"],
        "illustrative_pair_rule": "p03 and p15 at seed101 selected before outputs, all results retained separately",
        "figures": sorted(p.name for p in OUT.iterdir() if p.suffix in (".png", ".pdf")),
        "figure_sha256": {
            p.name: digest(p)
            for p in sorted(OUT.iterdir())
            if p.is_file() and p.suffix in (".png", ".pdf")
        },
    }
    (OUT / "figure_provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
