"""Copy the validation-selected trained adapter into the application's model folder."""

from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def select(run_name):
    run = ROOT / "05_Models/training_runs" / run_name
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["status"] != "TRAINING_COMPLETED"
        or summary["completed_steps"] != summary["max_steps"]
    ):
        raise ValueError("This run has not completed its declared training budget")
    source = ROOT / summary["best_adapter_path"] / "pytorch_lora_weights.safetensors"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = ROOT / "05_Models/cat_lora"
    target.mkdir(parents=True, exist_ok=True)
    existing = target / source.name
    if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() != digest:
        raise FileExistsError(
            "A different published adapter already exists; archive it before changing release identity"
        )
    shutil.copy2(source, existing)
    for source_path, name in [
        (
            ROOT / "05_Models/licenses/SD_CreativeML_OpenRAIL_M_LICENSE.txt",
            "LICENSE_BASE_MODEL.txt",
        ),
        (ROOT / "04_Data/licenses/AFHQ_CC_BY_NC_4_LICENSE.txt", "LICENSE_TRAINING_DATA.txt"),
    ]:
        shutil.copy2(source_path, target / name)
    provenance = {
        "adapter_sha256": digest,
        "adapter_bytes": existing.stat().st_size,
        "selected_run": run_name,
        "selection_rule": "Lowest fixed validation denoising MSE among trained checkpoints; no final prompt/test results consulted",
        "selected_step": summary["best_step"],
        "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "base_revision": summary["base_revision"],
        "training_summary": summary,
    }
    (target / "ADAPTER_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    text = f"""# CatStudio AFHQ LoRA — model card

## Intended use
Noncommercial educational/research cat-portrait generation. Input: an English cat portrait description and a random seed. Output: one 512 × 512 RGB image. The adapter requires the pinned SD1.5 base model; it cannot generate images on its own. Retain the upstream safety checker.

## Training provenance
- Base: `{provenance['base_model']}` at `{provenance['base_revision']}`.
- Adapter SHA256: `{digest}`; {existing.stat().st_size:,} bytes.
- Original data: AFHQ v1 cats, 5,653 photos. Training: 4,631; validation: 515; final reference/test: 500; excluded training neighbors of test: 7.
- Automatic local BLIP captions were conservatively cleaned to cat/coat descriptors; one contradictory sampled caption was replaced by a generic cat caption after an AI-assisted audit. Captions are weak labels, not human ground truth.
- UNet attention LoRA rank {summary['rank']}; {summary['trainable_parameters']:,} trainable parameters; frozen VAE, CLIP text encoder and base UNet weights.
- Actual optimizer updates: {summary['completed_steps']:,}; effective batch {summary['effective_batch_size']}; photo exposures {summary['image_exposures']:,}; effective epochs {summary['effective_epochs']:.3f}.
- Selected checkpoint: step {summary['best_step']}; fixed validation noise-prediction MSE {summary['best_validation_mse']:.6f}; initial base MSE {summary['initial_validation_mse']:.6f}.
- Hardware: {summary['device']}; training duration {summary['elapsed_seconds']/60:.1f} minutes, excluding data/caption/feature preprocessing.
- Training source SHA256: `{summary['training_source_sha256']}`. Full source snapshot, logs and resume state are in the training-run folder.

## Evaluation and limitations
The final frozen comparison uses 16 prompts × 4 seeds for both base and adapter under identical inference settings. See `06_Results/evaluation/summary.json` and all paired images after evaluation completes. Lower validation denoising loss does not guarantee better image quality. Report regressions as well as gains.

Training captions contain no reliable pose/background labels, so those capabilities mainly come from the pretrained model. Attribute descriptions may be incomplete. Generated anatomy, pose and requested colors can be wrong. A small cat dataset and brief fine-tuning budget do not create a general-purpose image model. Nearest-training-image screening cannot prove absence of memorization.

## Sources and license conditions
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. https://arxiv.org/abs/2112.10752
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
- Choi et al. (2020). StarGAN v2: Diverse Image Synthesis for Multiple Domains. https://github.com/clovaai/stargan-v2
- SD1.5 model card: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- BLIP: https://huggingface.co/Salesforce/blip-image-captioning-base

The base model is governed by CreativeML OpenRAIL-M. AFHQ is CC BY-NC 4.0 (NAVER Corporation); this project release is limited to noncommercial educational/research use and retains attribution and notice of changes. Complete source license texts accompany this adapter. No paid image API was used. Code and writing were developed with AI assistance and checked against actual execution evidence.
"""
    (target / "MODEL_CARD.md").write_text(text, encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_name")
    select(p.parse_args().run_name)
