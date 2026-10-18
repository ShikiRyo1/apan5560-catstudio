"""Regression checks for stale evaluation artifacts; no model inference occurs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_models.py"
    spec = importlib.util.spec_from_file_location("catstudio_evaluation_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PLAN", tmp_path / "absent_external_plan.json")
    base = tmp_path / "05_Models" / "base_sd15"
    base.mkdir(parents=True)
    (base / "PROVENANCE.json").write_text(json.dumps({"revision": "test-base-revision"}))
    adapter = tmp_path / "05_Models" / "cat_lora" / "pytorch_lora_weights.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"test-only adapter identity marker")
    return module, tmp_path, adapter


def metadata_for(module, path, adapter):
    prompt = {"id": "p01", "prompt": "a cat photograph"}
    plan = {"num_steps": 20, "guidance_scale": 7.5}
    metadata = {
        "model_choice": "lora", "prompt": prompt["prompt"], "prompt_id": "p01",
        "seed": 101, "num_steps": 20, "guidance_scale": 7.5,
        "width": 512, "height": 512, "adapter_scale": 1.0,
        "scheduler": "DPMSolverMultistepScheduler", "use_karras_sigmas": True,
        "negative_prompt": None, "safety_checker_enabled": True,
        "evaluation_plan_sha256": module.EXPECTED_PLAN_SHA256,
        "base_revision": "test-base-revision",
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return metadata, prompt, plan


def test_shipped_plan_works_without_external_results_folder(evaluation):
    module, *_ = evaluation
    assert len(module.plan()["prompts"]) == 16
    assert len(module.plan()["seeds"]) == 4


def test_unchanged_record_passes_but_modified_image_fails(evaluation):
    module, root, adapter = evaluation
    image = root / "example.png"
    image.write_bytes(b"test-only image identity marker")
    metadata, prompt, plan = metadata_for(module, image, adapter)
    module.verify_generation(image, metadata, "lora", prompt, 101, plan)
    image.write_bytes(b"replacement image, same filename")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        module.verify_generation(image, metadata, "lora", prompt, 101, plan)


def test_new_adapter_at_same_path_rejects_cached_old_output(evaluation):
    module, root, adapter = evaluation
    image = root / "example.png"
    image.write_bytes(b"test-only image identity marker")
    metadata, prompt, plan = metadata_for(module, image, adapter)
    adapter.write_bytes(b"another trained checkpoint, same filename")
    with pytest.raises(ValueError, match="adapter_sha256"):
        module.verify_generation(image, metadata, "lora", prompt, 101, plan)


def test_wrong_generation_settings_reject_cached_output(evaluation):
    module, root, adapter = evaluation
    image = root / "example.png"
    image.write_bytes(b"test-only image identity marker")
    metadata, prompt, plan = metadata_for(module, image, adapter)
    metadata["seed"] = 999
    with pytest.raises(ValueError, match="seed"):
        module.verify_generation(image, metadata, "lora", prompt, 101, plan)
