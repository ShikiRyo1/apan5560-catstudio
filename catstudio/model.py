"""Local Stable Diffusion inference with optional, explicitly selected LoRA."""

from __future__ import annotations
import hashlib, json, threading, time
from pathlib import Path
from typing import Any


class PortraitEngine:
    """Lazily load one local pipeline; never substitute a mock or download silently."""

    def __init__(self, base_model_dir, adapter_dir=None, device="auto"):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        self.base_model_dir = Path(base_model_dir)
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.requested_device = device
        self.device = None
        self.pipeline = None
        self._adapter_loaded = False
        self._adapter_sha256 = None
        self._base_revision = None
        self._lock = threading.RLock()

    def status(self) -> dict[str, Any]:
        required = [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "safety_checker/model.safetensors",
            "tokenizer/vocab.json",
            "scheduler/scheduler_config.json",
        ]
        base = all((self.base_model_dir / f).is_file() for f in required)
        adapter = bool(
            self.adapter_dir and (self.adapter_dir / "pytorch_lora_weights.safetensors").is_file()
        )
        return {
            "loaded": self.pipeline is not None,
            "device": self.device or self.requested_device,
            "base_model_available": base,
            "adapter_available": adapter,
            "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "adapter_name": "CatStudio AFHQ LoRA" if adapter else None,
            "safety_checker_enabled": True,
            "downloads_automatic": False,
        }

    def _load(self):
        if self.pipeline is not None:
            return
        if not self.status()["base_model_available"]:
            raise FileNotFoundError(
                "Base model is absent. Run scripts/download_assets.py sd15 first."
            )
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.requested_device == "auto"
            else self.requested_device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if self.device == "cpu":
            torch.set_num_threads(min(8, torch.get_num_threads()))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            str(self.base_model_dir), torch_dtype=dtype, local_files_only=True, use_safetensors=True
        )
        if pipe.safety_checker is None:
            raise RuntimeError("Upstream safety checker is required but missing.")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.to(self.device)
        pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)
        provenance = self.base_model_dir / "PROVENANCE.json"
        if provenance.exists():
            self._base_revision = json.loads(provenance.read_text())["revision"]
        self.pipeline = pipe

    def generate(
        self,
        prompt,
        seed=42,
        num_steps=20,
        guidance_scale=7.5,
        adapter_scale=1.0,
        model_choice="lora",
        negative_prompt=None,
    ):
        """Return a 512x512 PIL image and the actual generation settings/timing."""
        if model_choice not in ("base", "lora"):
            raise ValueError("model_choice must be base or lora")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        if not 1 <= int(num_steps) <= 60:
            raise ValueError("num_steps must be between 1 and 60")
        if not 0 <= int(seed) < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit integer")
        if not 0 <= float(adapter_scale) <= 1.5:
            raise ValueError("adapter_scale must be between 0 and 1.5")
        if not 1 <= float(guidance_scale) <= 20:
            raise ValueError("guidance_scale must be between 1 and 20")
        if model_choice == "lora" and not self.status()["adapter_available"]:
            raise FileNotFoundError(
                "Trained CatStudio LoRA is absent; choose base explicitly or install the adapter."
            )
        with self._lock:
            import torch

            loading_start = time.perf_counter()
            self._load()
            load_seconds = time.perf_counter() - loading_start
            pipe = self.pipeline
            if model_choice == "lora":
                current_hash = hashlib.sha256(
                    (self.adapter_dir / "pytorch_lora_weights.safetensors").read_bytes()
                ).hexdigest()
                if self._adapter_loaded and current_hash != self._adapter_sha256:
                    raise RuntimeError(
                        "Adapter file changed after loading; restart the application"
                    )
                if not self._adapter_loaded:
                    pipe.load_lora_weights(
                        str(self.adapter_dir),
                        weight_name="pytorch_lora_weights.safetensors",
                        adapter_name="catstudio",
                        local_files_only=True,
                    )
                    self._adapter_sha256 = current_hash
                    self._adapter_loaded = True
                pipe.enable_lora()
                pipe.set_adapters("catstudio", adapter_weights=float(adapter_scale))
            elif self._adapter_loaded:
                pipe.disable_lora()
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            token_count = len(pipe.tokenizer(prompt.strip()).input_ids)
            start = time.perf_counter()
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt.strip(),
                    negative_prompt=negative_prompt,
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                    num_inference_steps=int(num_steps),
                    guidance_scale=float(guidance_scale),
                    height=512,
                    width=512,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            metadata = {
                "actual_model": "sd15_catstudio_lora" if model_choice == "lora" else "sd15_base",
                "model_choice": model_choice,
                "model_id": self.status()["model_id"],
                "prompt": prompt.strip(),
                "negative_prompt": negative_prompt,
                "seed": int(seed),
                "num_steps": int(num_steps),
                "guidance_scale": float(guidance_scale),
                "adapter_scale": float(adapter_scale) if model_choice == "lora" else 0.0,
                "scheduler": type(pipe.scheduler).__name__,
                "use_karras_sigmas": True,
                "width": 512,
                "height": 512,
                "device": self.device,
                "latency_seconds": round(time.perf_counter() - start, 4),
                "load_seconds": round(load_seconds, 4),
                "peak_gpu_allocated_mb": (
                    round(torch.cuda.max_memory_allocated() / 1024**2, 3)
                    if self.device == "cuda"
                    else None
                ),
                "peak_gpu_reserved_mb": (
                    round(torch.cuda.max_memory_reserved() / 1024**2, 3)
                    if self.device == "cuda"
                    else None
                ),
                "nsfw_content_detected": bool(
                    result.nsfw_content_detected and result.nsfw_content_detected[0]
                ),
                "safety_checker_enabled": True,
                "prompt_token_count": token_count,
                "prompt_max_tokens": pipe.tokenizer.model_max_length,
                "prompt_was_truncated": token_count > pipe.tokenizer.model_max_length,
            }
            if self._base_revision:
                metadata["base_revision"] = self._base_revision
            if model_choice == "lora":
                metadata["adapter_sha256"] = self._adapter_sha256
            return result.images[0], metadata
