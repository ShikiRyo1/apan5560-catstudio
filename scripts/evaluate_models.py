"""Frozen paired text-to-image comparison; all metrics are explicitly proxies."""

from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catstudio.model import PortraitEngine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "06_Results/evaluation"
PLAN = ROOT / "06_Results/evaluation_plan.json"
SHIPPED_PLAN = Path(__file__).resolve().parents[1] / "config/evaluation_plan.json"
EXPECTED_PLAN_SHA256 = "cf50a96e635f418cdc17d20b6ef8ebf1e537015a0b47a348c44816e4dda295be"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def plan():
    raw = (PLAN if PLAN.exists() else SHIPPED_PLAN).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PLAN_SHA256:
        raise ValueError(
            "Frozen evaluation plan changed. Audit and explicitly update fingerprint before a new experiment."
        )
    return json.loads(raw)


def verify_generation(path, metadata, choice, prompt, seed, p):
    expected = {
        "model_choice": choice,
        "prompt": prompt["prompt"],
        "prompt_id": prompt["id"],
        "seed": seed,
        "num_steps": p["num_steps"],
        "guidance_scale": p["guidance_scale"],
        "width": 512,
        "height": 512,
        "adapter_scale": 1.0 if choice == "lora" else 0.0,
        "scheduler": "DPMSolverMultistepScheduler",
        "use_karras_sigmas": True,
        "negative_prompt": None,
        "safety_checker_enabled": True,
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "base_revision": json.loads((ROOT / "05_Models/base_sd15/PROVENANCE.json").read_text())[
            "revision"
        ],
    }
    if choice == "lora":
        expected["adapter_sha256"] = hashlib.sha256(
            (ROOT / "05_Models/cat_lora/pytorch_lora_weights.safetensors").read_bytes()
        ).hexdigest()
    mismatches = {k: (metadata.get(k), v) for k, v in expected.items() if metadata.get(k) != v}
    if mismatches:
        raise ValueError(f"Stale or mismatched generated image metadata: {path}: {mismatches}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("image_sha256"):
        raise ValueError(f"Image hash mismatch: {path}")


def generate(choice):
    p = plan()
    out = OUT / "images" / choice
    out.mkdir(parents=True, exist_ok=True)
    engine = PortraitEngine(ROOT / "05_Models/base_sd15", ROOT / "05_Models/cat_lora", "cuda")
    for prompt in p["prompts"]:
        for seed in p["seeds"]:
            name = f'{prompt["id"]}_s{seed}'
            image_file = out / (name + ".png")
            meta_file = out / (name + ".json")
            if image_file.exists() and meta_file.exists():
                verify_generation(
                    image_file, json.loads(meta_file.read_text()), choice, prompt, seed, p
                )
                continue
            image, meta = engine.generate(
                prompt["prompt"], seed, p["num_steps"], p["guidance_scale"], 1.0, choice
            )
            image.save(image_file)
            meta.update(
                {
                    "prompt_id": prompt["id"],
                    "file": image_file.relative_to(ROOT).as_posix(),
                    "image_sha256": hashlib.sha256(image_file.read_bytes()).hexdigest(),
                    "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
                }
            )
            save(meta_file, meta)
            print(json.dumps(meta), flush=True)
    contact_sheets()


def contact_sheets():
    p = plan()
    folder = OUT / "contact_sheets"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in p["seeds"]:
        for part in range(4):
            canvas = Image.new("RGB", (1100, 1240), "white")
            d = ImageDraw.Draw(canvas)
            for row, pr in enumerate(p["prompts"][part * 4 : part * 4 + 4]):
                for col, choice in enumerate(("base", "lora")):
                    x = col * 550
                    y = row * 310
                    file = OUT / "images" / choice / f'{pr["id"]}_s{seed}.png'
                    if file.exists():
                        canvas.paste(
                            Image.open(file).convert("RGB").resize((280, 280)), (x, y + 25)
                        )
                    d.text(
                        (x + 5, y + 5), f'{choice.upper()} | {pr["id"]} | seed {seed}', fill="black"
                    )
                    import textwrap

                    d.text(
                        (x + 290, y + 55), "\n".join(textwrap.wrap(pr["prompt"], 32)), fill="black"
                    )
            canvas.save(folder / f"paired_seed{seed}_page{part+1}.jpg", quality=93)


def paired_bootstrap(values, seed=20261018):
    # values[prompt, model]; seeds have already been averaged within prompt.
    rng = np.random.default_rng(seed)
    delta = values[:, 1] - values[:, 0]
    samples = rng.integers(0, len(delta), (5000, len(delta)))
    boot = delta[samples].mean(axis=1)
    return {
        "base_mean": float(values[:, 0].mean()),
        "lora_mean": float(values[:, 1].mean()),
        "paired_delta_lora_minus_base": float(delta.mean()),
        "paired_prompt_cluster_bootstrap_95pct_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "prompt_clusters": len(delta),
        "bootstrap_replicates": 5000,
    }


def metrics():
    import torch, imagehash
    from transformers import CLIPProcessor, CLIPModel

    torch.set_num_threads(6)
    p = plan()
    device = "cuda"
    checkpoint = ROOT / "05_Models/clip_evaluator"
    processor = CLIPProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = CLIPModel.from_pretrained(checkpoint, local_files_only=True).to(device).eval()

    def image_features(paths):
        result = []
        for start in range(0, len(paths), 32):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + 32]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            result.append(features.cpu().numpy())
            if start % 512 == 0:
                print(f"CLIP images {start}/{len(paths)}", flush=True)
        return np.concatenate(result)

    def text_features(texts):
        inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.inference_mode():
            f = model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    prompts = p["prompts"]
    seeds = p["seeds"]
    meta = []
    paths = []
    for choice in ("base", "lora"):
        for pr in prompts:
            for seed in seeds:
                path = OUT / "images" / choice / f'{pr["id"]}_s{seed}.png'
                mfile = path.with_suffix(".json")
                if not path.exists() or not mfile.exists():
                    raise FileNotFoundError(path)
                m = json.loads(mfile.read_text())
                verify_generation(path, m, choice, pr, seed, p)
                m.update(
                    {
                        "requested_color": pr["color"],
                        "requested_pattern": pr["pattern"],
                        "requested_pose": pr["pose"],
                    }
                )
                meta.append(m)
                paths.append(path)
    feats = image_features(paths)
    np.save(OUT / "generated_clip_features.npy", feats)
    prompt_features = text_features([pr["prompt"] for pr in prompts])
    colors = ["black", "white", "orange", "gray", "black and white", "orange and white", "brown"]
    patterns = ["solid", "tabby", "bicolor"]
    poses = ["front", "three-quarter", "profile"]
    color_f = text_features([f"a close-up photograph of a {c} cat" for c in colors])
    pattern_f = text_features(
        [
            "a cat with solid unpatterned fur",
            "a tabby cat with striped fur",
            "a bicolor cat with two colors of fur",
        ]
    )
    pose_f = text_features(
        [
            "a cat face viewed from the front",
            "a cat face in three-quarter view",
            "a cat face in side profile",
        ]
    )
    records = [
        json.loads(x) for x in (ROOT / "04_Data/metadata/images.jsonl").read_text().splitlines()
    ]
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    reference_cache = OUT / "reference_clip_features.npz"
    clip_revision = json.loads((checkpoint / "PROVENANCE.json").read_text())["revision"]
    reference_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "clip_revision": clip_revision,
                "ordered_records": [
                    {k: r[k] for k in ("id", "sha256", "split")} for r in train + test
                ],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if reference_cache.exists():
        cached = np.load(reference_cache)
        train_f = cached["train"]
        test_f = cached["test"]
        if str(cached["fingerprint"]) != reference_fingerprint:
            raise ValueError(
                "Stale reference CLIP feature cache: image order/content/splits or evaluator revision changed"
            )
        assert len(train_f) == len(train) and len(test_f) == len(test)
    else:
        train_f = image_features([ROOT / r["path"] for r in train])
        test_f = image_features([ROOT / r["path"] for r in test])
        np.savez(reference_cache, train=train_f, test=test_f, fingerprint=reference_fingerprint)
    train_hash = [int(r["phash"], 16) for r in train]
    n = len(prompts) * len(seeds)
    for i, (m, path) in enumerate(zip(meta, paths)):
        pr_idx = (i % n) // len(seeds)
        m["clip_text_cosine"] = float(feats[i] @ prompt_features[pr_idx])
        m["proxy_predicted_color"] = colors[int(np.argmax(feats[i] @ color_f.T))]
        m["proxy_predicted_pattern"] = patterns[int(np.argmax(feats[i] @ pattern_f.T))]
        m["proxy_predicted_pose"] = poses[int(np.argmax(feats[i] @ pose_f.T))]
        for attr in ("color", "pattern", "pose"):
            m[f"{attr}_proxy_agreement"] = int(
                m[f"proxy_predicted_{attr}"] == m[f"requested_{attr}"]
            )
        similarities = feats[i] @ train_f.T
        j = int(np.argmax(similarities))
        m["nearest_train_clip_cosine"] = float(similarities[j])
        m["nearest_train_id"] = train[j]["id"]
        m["nearest_train_path"] = train[j]["path"]
        ph = int(str(imagehash.phash(Image.open(path))), 16)
        distances = [(ph ^ h).bit_count() for h in train_hash]
        k = int(np.argmin(distances))
        m["nearest_train_phash_distance"] = distances[k]
        m["nearest_train_phash_id"] = train[k]["id"]
    save(OUT / "per_image_metrics.json", meta)
    summary = {
        "evaluation_plan_sha256": EXPECTED_PLAN_SHA256,
        "per_image_metrics_sha256": hashlib.sha256(
            (OUT / "per_image_metrics.json").read_bytes()
        ).hexdigest(),
        "generated_clip_features_sha256": hashlib.sha256(
            (OUT / "generated_clip_features.npy").read_bytes()
        ).hexdigest(),
        "base_revision": meta[0]["base_revision"],
        "adapter_sha256": meta[n]["adapter_sha256"],
        "image_pairs": n,
        "prompt_clusters": len(prompts),
        "seeds_per_prompt": len(seeds),
        "clip_evaluator": "openai/clip-vit-base-patch32",
        "clip_revision": json.loads((checkpoint / "PROVENANCE.json").read_text())["revision"],
        "metrics": {},
        "limitations": [
            "CLIP similarity and attribute agreement are automated proxies, not human quality or factual accuracy.",
            "Small fixed prompt grid is a course experiment, not a broad population benchmark.",
            "Nearest-image screening cannot prove absence of memorization.",
            "No target photo exists for an open-ended prompt; SSIM/PSNR are deliberately not used.",
            "Independent CLIP evaluator may still share source-data biases with the generator.",
            "The 500 reference photos were held out from this project's LoRA training and selection; overlap with SD1.5 pretraining data is unknown.",
            "GPU memory reports PyTorch peak allocated/reserved memory in this process, not total GPU usage by all applications.",
        ],
    }
    for key in (
        "clip_text_cosine",
        "color_proxy_agreement",
        "pattern_proxy_agreement",
        "pose_proxy_agreement",
        "nearest_train_clip_cosine",
        "nearest_train_phash_distance",
    ):
        a = np.array([m[key] for m in meta]).reshape(2, len(prompts), len(seeds)).mean(axis=2).T
        summary["metrics"][key] = paired_bootstrap(a)
    diversity = []
    for choice in range(2):
        scores = []
        for j in range(len(prompts)):
            f = feats[choice * n + j * len(seeds) : choice * n + (j + 1) * len(seeds)]
            dist = 1 - f @ f.T
            scores.append(float(dist[np.triu_indices(len(seeds), 1)].mean()))
        diversity.append(scores)
    summary["metrics"]["within_prompt_clip_cosine_distance_diversity"] = paired_bootstrap(
        np.array(diversity).T
    )
    for model_idx, choice in enumerate(("base", "lora")):
        rows = meta[model_idx * n : (model_idx + 1) * n]
        summary[choice] = {
            "nsfw_flagged": sum(m["nsfw_content_detected"] for m in rows),
            "mean_generation_seconds_excluding_first": float(
                np.mean([m["latency_seconds"] for m in rows[1:]])
            ),
            "all_generation_seconds": sum(m["latency_seconds"] for m in rows),
            "peak_gpu_allocated_mb": max(m["peak_gpu_allocated_mb"] for m in rows),
            "peak_gpu_reserved_mb": max(m["peak_gpu_reserved_mb"] for m in rows),
            "nearest_train_phash_min": min(m["nearest_train_phash_distance"] for m in rows),
        }
        # Exploratory distribution diagnostic on the reserved 500 reference cats.
        f = feats[model_idx * n : (model_idx + 1) * n]
        kxx = (1 + f @ f.T) ** 3
        kyy = (1 + test_f @ test_f.T) ** 3
        kxy = (1 + f @ test_f.T) ** 3
        mmd = (
            (kxx.sum() - np.trace(kxx)) / (len(f) * (len(f) - 1))
            + (kyy.sum() - np.trace(kyy)) / (len(test_f) * (len(test_f) - 1))
            - 2 * kxy.mean()
        )
        summary[choice]["exploratory_clip_polynomial_mmd2_vs_500_test"] = float(mmd)
    summary["exploratory_mmd_note"] = (
        "Unbiased squared MMD with kernel (1 + cosine)^3 on normalized CLIP features; may be negative in finite samples. Exploratory diagnostic added before output generation; not standard Inception KID/FID and not used to select checkpoint."
    )
    save(OUT / "summary.json", summary)
    nearest_sheet(meta)
    print(json.dumps(summary, indent=2), flush=True)


def nearest_sheet(meta):
    out = OUT / "contact_sheets"
    out.mkdir(parents=True, exist_ok=True)
    for choice in ("base", "lora"):
        rows = sorted(
            [m for m in meta if m["model_choice"] == choice],
            key=lambda m: m["nearest_train_clip_cosine"],
            reverse=True,
        )[:8]
        canvas = Image.new("RGB", (1100, 1200), "white")
        d = ImageDraw.Draw(canvas)
        for j, m in enumerate(rows):
            x = (j % 2) * 550
            y = (j // 2) * 300
            canvas.paste(
                Image.open(ROOT / m["file"]).convert("RGB").resize((250, 250)), (x, y + 30)
            )
            canvas.paste(
                Image.open(ROOT / m["nearest_train_path"]).convert("RGB").resize((250, 250)),
                (x + 265, y + 30),
            )
            d.text(
                (x, y + 4),
                f'{m["prompt_id"]}/{m["seed"]} generated | nearest training, cosine={m["nearest_train_clip_cosine"]:.3f}',
                fill="black",
            )
        canvas.save(out / f"nearest_training_{choice}.jpg", quality=93)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["base", "lora", "metrics", "sheets"])
    args = parser.parse_args()
    if args.phase in ("base", "lora"):
        generate(args.phase)
    elif args.phase == "metrics":
        metrics()
    else:
        contact_sheets()
