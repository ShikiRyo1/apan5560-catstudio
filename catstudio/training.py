"""Memory-conscious SD1.5 LoRA training, validation, and frozen-encoder caching.

Follows the Hugging Face Diffusers text-to-image LoRA recipe: frozen VAE/CLIP,
DDPM noise-prediction loss, PEFT adapters on UNet attention projections only.
"""

from __future__ import annotations
import gc, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "05_Models/hf_cache"))


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_features():
    """Cache frozen encoder outputs; posterior noise is resampled during training."""
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    caption_file = ROOT / "04_Data/metadata/captions.jsonl"
    caption_sha = hashlib.sha256(caption_file.read_bytes()).hexdigest()
    records = [json.loads(x) for x in caption_file.read_text(encoding="utf-8").splitlines()]
    records = sorted(records, key=lambda x: x["id"])
    path = ROOT / "04_Data/features_512"
    path.mkdir(parents=True, exist_ok=True)
    if (path / "complete.json").exists():
        if json.loads((path / "complete.json").read_text())["caption_file_sha256"] != caption_sha:
            raise ValueError(
                "Captions changed after caching; archive old features before explicitly rebuilding."
            )
        print("Cache already complete and caption hash matches", flush=True)
        return
    model_path = ROOT / "05_Models/base_sd15"
    device = "cuda"
    tokenizer = CLIPTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
    encoder = (
        CLIPTextModel.from_pretrained(
            model_path / "text_encoder", local_files_only=True, torch_dtype=torch.float16
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    vae = (
        AutoencoderKL.from_pretrained(
            model_path / "vae", local_files_only=True, torch_dtype=torch.float32
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    n = len(records)
    mean = np.lib.format.open_memmap(
        path / "latent_mean.npy", mode="w+", dtype=np.float16, shape=(n, 4, 64, 64)
    )
    logvar = np.lib.format.open_memmap(
        path / "latent_logvar.npy", mode="w+", dtype=np.float16, shape=(n, 4, 64, 64)
    )
    text = np.lib.format.open_memmap(
        path / "text_embeddings.npy", mode="w+", dtype=np.float16, shape=(n, 77, 768)
    )
    started = time.time()
    for start in range(0, n, 2):
        batch = records[start : start + 2]
        images = np.stack(
            [
                np.asarray(
                    Image.open(ROOT / r["path"])
                    .convert("RGB")
                    .resize((512, 512), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
                / 127.5
                - 1
                for r in batch
            ]
        )
        pixels = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
        ids = tokenizer(
            [r["caption"] for r in batch],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).input_ids.to(device)
        with torch.inference_mode():
            posterior = vae.encode(pixels).latent_dist
            emb = encoder(ids)[0]
        mean[start : start + len(batch)] = posterior.mean.cpu().numpy()
        logvar[start : start + len(batch)] = posterior.logvar.cpu().numpy()
        text[start : start + len(batch)] = emb.cpu().numpy()
        if start % 100 == 0:
            print(f"cached {start+len(batch)}/{n}, seconds={time.time()-started:.1f}", flush=True)
    mean.flush()
    logvar.flush()
    text.flush()
    json_save(path / "records.json", records)
    json_save(
        path / "complete.json",
        {
            "count": n,
            "resolution": 512,
            "caption_file_sha256": caption_sha,
            "latent_scaling_factor": float(vae.config.scaling_factor),
            "vae_sampling": "posterior mean/logvar cached; fresh posterior sample at each training exposure",
            "augmentation": "none: avoids contradicting directional caption text",
            "elapsed_seconds": time.time() - started,
        },
    )


def train(
    run_name="full",
    max_steps=3000,
    accumulation=4,
    rank=8,
    lr=1e-4,
    checkpoint_every=500,
    resume=None,
):
    if max_steps < 1 or accumulation < 1 or checkpoint_every < 1 or not 1 <= rank <= 64:
        raise ValueError("Steps, accumulation and checkpoint interval must be positive; rank 1..64")
    if not 0 < lr <= 0.01:
        raise ValueError("Learning rate must be in (0, 0.01]")
    import torch
    from torch.utils.data import Dataset, DataLoader
    from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

    seed = 20261018
    seed_all(seed)
    torch.set_num_threads(6)
    if not torch.cuda.is_available():
        raise RuntimeError("Full training requires CUDA; no silent CPU fallback.")
    torch.backends.cuda.matmul.allow_tf32 = True
    cache = ROOT / "04_Data/features_512"
    assert (cache / "complete.json").exists(), "Run cache_features first"
    cache_info = json.loads((cache / "complete.json").read_text())
    if (
        hashlib.sha256((ROOT / "04_Data/metadata/captions.jsonl").read_bytes()).hexdigest()
        != cache_info["caption_file_sha256"]
    ):
        raise ValueError("Training captions differ from frozen feature cache")
    latent_scale = cache_info["latent_scaling_factor"]
    records = json.loads((cache / "records.json").read_text(encoding="utf-8"))
    means = np.load(cache / "latent_mean.npy", mmap_mode="r")
    logvars = np.load(cache / "latent_logvar.npy", mmap_mode="r")
    texts = np.load(cache / "text_embeddings.npy", mmap_mode="r")
    if len(records) != cache_info["count"] or len({r["id"] for r in records}) != len(records):
        raise ValueError("Feature cache record count or unique IDs are inconsistent")
    if means.shape != (len(records), 4, 64, 64) or logvars.shape != means.shape:
        raise ValueError("VAE feature cache has an unexpected shape")
    if texts.shape != (len(records), 77, 768) or not 0 < latent_scale < 1:
        raise ValueError("Text features or latent scaling factor are invalid")
    if any(r["split"] not in ("train", "validation") for r in records):
        raise ValueError("Test or excluded images entered the training feature cache")
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]
    val_idx = [i for i, r in enumerate(records) if r["split"] == "validation"]
    if not train_idx or not val_idx:
        raise ValueError("Training and validation partitions must both be nonempty")
    if {records[i]["duplicate_group"] for i in train_idx} & {
        records[i]["duplicate_group"] for i in val_idx
    }:
        raise ValueError("A duplicate group crosses training and validation")
    rng = random.Random(741)
    rng.shuffle(val_idx)
    val_idx = val_idx[:64]

    class Features(Dataset):
        def __len__(self):
            return len(train_idx)

        def __getitem__(self, j):
            i = train_idx[j]
            return tuple(
                torch.from_numpy(np.array(a[i], copy=True)) for a in (means, logvars, texts)
            )

    dataset = Features()
    base = ROOT / "05_Models/base_sd15"
    out = ROOT / "05_Models/training_runs" / run_name
    if (out / "config.json").exists() and not resume:
        raise FileExistsError("Run name already exists; choose a new name or explicitly resume")
    out.mkdir(parents=True, exist_ok=True)
    unet = (
        UNet2DConditionModel.from_pretrained(
            base / "unet", torch_dtype=torch.float16, local_files_only=True
        )
        .to("cuda")
        .requires_grad_(False)
    )
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    unet.enable_gradient_checkpointing()
    scheduler = DDPMScheduler.from_pretrained(base / "scheduler", local_files_only=True)
    assert scheduler.config.prediction_type == "epsilon"
    params = [p for p in unet.parameters() if p.requires_grad]
    warmup = min(100, max_steps // 10)
    opt = torch.optim.AdamW(
        params, lr=lr / max(1, warmup), weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    scale = torch.amp.GradScaler("cuda")
    config = {
        "run_name": run_name,
        "seed": seed,
        "resolution": 512,
        "max_steps": max_steps,
        "gradient_accumulation": accumulation,
        "micro_batch_size": 1,
        "effective_batch_size": accumulation,
        "rank": rank,
        "learning_rate": lr,
        "lr_schedule": "linear warmup then cosine",
        "warmup_steps": warmup,
        "train_images": len(train_idx),
        "validation_denoising_images": len(val_idx),
        "validation_seed": 741,
        "validation_image_ids": [records[i]["id"] for i in val_idx],
        "caption_file_sha256": cache_info["caption_file_sha256"],
        "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trainable_parameters": sum(p.numel() for p in params),
        "frozen_parameters": sum(p.numel() for p in unet.parameters() if not p.requires_grad),
        "loss": "epsilon prediction MSE, uniformly sampled DDPM timestep",
        "dtype": "frozen fp16; LoRA fp32; autocast fp16",
        "optimizer": "AdamW",
        "augmentation": "none",
        "checkpoint_selection": "lowest fixed validation noise-prediction MSE; test data not consulted",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "base_revision": json.loads((base / "PROVENANCE.json").read_text())["revision"],
    }
    if not resume:
        json_save(out / "config.json", config)
        (out / "training_source_snapshot.py").write_bytes(Path(__file__).read_bytes())

    def validate():
        unet.eval()
        generator = torch.Generator(device="cuda").manual_seed(741)
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in val_idx:
                mean = torch.from_numpy(np.array(means[i], copy=True))[None].to("cuda")
                lv = torch.from_numpy(np.array(logvars[i], copy=True))[None].to("cuda")
                latent = (
                    mean
                    + torch.exp(0.5 * lv)
                    * torch.randn(mean.shape, generator=generator, device="cuda", dtype=mean.dtype)
                ) * latent_scale
                noise = torch.randn(
                    latent.shape, generator=generator, device="cuda", dtype=latent.dtype
                )
                t = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (1,),
                    generator=generator,
                    device="cuda",
                )
                emb = torch.from_numpy(np.array(texts[i], copy=True))[None].to("cuda")
                pred = unet(scheduler.add_noise(latent, noise, t), t, emb).sample
                losses.append(torch.nn.functional.mse_loss(pred.float(), noise.float()).item())
        unet.train()
        return float(np.mean(losses))

    def save_adapter(folder):
        folder.mkdir(parents=True, exist_ok=True)
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            str(folder), unet_lora_layers=state, safe_serialization=True
        )

    print(json.dumps(config), flush=True)
    started = time.time()
    best = math.inf
    best_step = None
    micro = 0
    step = 0
    step_losses = []
    if resume:
        # Only load a checkpoint produced by this project, never an untrusted pickle.
        state = torch.load(
            Path(resume) / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        for key in (
            "seed",
            "max_steps",
            "rank",
            "gradient_accumulation",
            "learning_rate",
            "train_images",
            "caption_file_sha256",
            "base_revision",
            "validation_image_ids",
            "training_source_sha256",
            "trainable_parameters",
        ):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        set_peft_model_state_dict(unet, state["adapter_state"])
        opt.load_state_dict(state["optimizer"])
        scale.load_state_dict(state["scaler"])
        for optimizer_state in opt.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value) and key != "step":
                    optimizer_state[key] = value.to("cuda")
        step = state["step"]
        micro = state["micro"]
        best = state["best"]
        best_step = state["best_step"]
        history = state["history"]
        initial_val = state["initial_validation_mse"]
        started -= state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        # Preserve existing evidence until compatibility and restoration succeed.
        json_save(out / "config.json", config)
        (out / "training_source_snapshot.py").write_bytes(Path(__file__).read_bytes())
        print(
            f"Resumed optimizer, RNG, data position, scaler and adapter at step {step}", flush=True
        )
    else:
        initial_val = validate()
        print(f"Initial validation MSE: {initial_val}", flush=True)
        history = [
            {"step": 0, "validation_mse": initial_val, "elapsed_seconds": time.time() - started}
        ]
        json_save(out / "validation_history.json", history)
    unet.train()
    opt.zero_grad(set_to_none=True)
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    while step < max_steps:
        epoch = micro // len(dataset)
        offset = micro % len(dataset)
        order = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()[offset:]
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=order,
            num_workers=0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        for mean, lv, emb in loader:
            mean, lv, emb = (x.to("cuda", non_blocking=True) for x in (mean, lv, emb))
            latent = (mean + torch.exp(0.5 * lv) * torch.randn_like(mean)) * latent_scale
            noise = torch.randn_like(latent)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = unet(scheduler.add_noise(latent, noise, t), t, emb).sample
                loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scale.scale(loss / accumulation).backward()
            micro += 1
            step_losses.append(loss.item())
            if micro % accumulation:
                continue
            scale.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            scale.step(opt)
            scale.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            factor = (
                ((step + 1) / max(1, warmup))
                if step < warmup
                else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, max_steps - warmup)))
            )
            for group in opt.param_groups:
                group["lr"] = lr * factor
            row = {
                "step": step,
                "loss": float(np.mean(step_losses)),
                "lr": opt.param_groups[0]["lr"],
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                "image_exposures": micro,
            }
            step_losses = []
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            if step % checkpoint_every == 0 or step == max_steps:
                score = validate()
                folder = out / f"checkpoint-{step:05d}"
                save_adapter(folder)
                row = {
                    "step": step,
                    "validation_mse": score,
                    "elapsed_seconds": time.time() - started,
                    "adapter_path": str(folder.relative_to(ROOT)).replace("\\", "/"),
                }
                history.append(row)
                json_save(out / "validation_history.json", history)
                if score < best:
                    best, best_step = score, step
                torch.save(
                    {
                        "config": config,
                        "adapter_state": get_peft_model_state_dict(unet),
                        "optimizer": opt.state_dict(),
                        "scaler": scale.state_dict(),
                        "step": step,
                        "micro": micro,
                        "best": best,
                        "best_step": best_step,
                        "history": history,
                        "initial_validation_mse": initial_val,
                        "elapsed_seconds": time.time() - started,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(),
                    },
                    folder / "trainer_state.pt",
                )
                print("VALIDATION " + json.dumps(row), flush=True)
            if step >= max_steps:
                break
    log.close()
    save_adapter(out / "last")
    summary = {
        **config,
        "completed_steps": step,
        "image_exposures": micro,
        "effective_epochs": micro / len(train_idx),
        "elapsed_seconds": time.time() - started,
        "initial_validation_mse": initial_val,
        "best_validation_mse": best,
        "best_step": best_step,
        "best_adapter": f"checkpoint-{best_step:05d}",
        "best_adapter_path": next(
            row["adapter_path"] for row in history if row["step"] == best_step
        ),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "status": "TRAINING_COMPLETED",
    }
    json_save(out / "summary.json", summary)
    print("COMPLETED " + json.dumps(summary), flush=True)
