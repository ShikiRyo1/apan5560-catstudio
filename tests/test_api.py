"""API/control-flow tests with an explicitly fake image engine.

These tests verify validation, failures, concurrency and transport. They do not
measure diffusion quality or demonstrate that the real model can run.
"""

import base64
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from catstudio.api import BASE_REQUIRED_FILES, Settings, create_app


class FakeEngine:
    device = "test-double"

    def __init__(self):
        self.calls = []

    def status(self):
        return {"loaded": bool(self.calls), "device": self.device}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return Image.new("RGB", (8, 8), (100, 120, 140)), {"test_double": True, **kwargs}


@pytest.fixture
def setup(tmp_path):
    config = Settings(tmp_path / "base", tmp_path / "adapter", "cpu")
    config.base_model_dir.mkdir()
    config.adapter_dir.mkdir()
    for name in BASE_REQUIRED_FILES:
        marker = config.base_model_dir / name
        marker.parent.mkdir(exist_ok=True, parents=True)
        marker.write_text("test-double marker, not weights")
    (config.adapter_dir / "pytorch_lora_weights.safetensors").write_text("test-double marker, not weights")
    engine = FakeEngine()
    app = create_app(config, lambda settings: engine)
    with TestClient(app) as client:
        yield client, app, config, engine


def test_health_does_not_claim_loaded_before_inference(setup):
    client, app, config, engine = setup
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["portrait"]["base_files_present"] is True
    assert response.json()["portrait"]["model_loaded"] is False
    assert engine.calls == []


def test_frontend_and_openapi_document_real_endpoint(setup):
    client, *_ = setup
    assert "CatStudio" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert "/portraits" in client.get("/openapi.json").json()["paths"]


def test_image_and_settings_roundtrip_using_test_double(setup):
    client, app, config, engine = setup
    result = client.post("/portraits", json={"prompt": "  A gray cat portrait  ", "seed": 123, "model_choice": "base"})
    assert result.status_code == 200
    assert engine.calls[0]["prompt"] == "A gray cat portrait"
    assert engine.calls[0]["seed"] == 123
    assert engine.calls[0]["model_choice"] == "base"
    image = Image.open(BytesIO(base64.b64decode(result.json()["image_base64"])))
    assert image.size == (8, 8)
    assert result.json()["metadata"]["test_double"] is True
    assert client.get("/health").json()["portrait"]["model_loaded"] is True


@pytest.mark.parametrize("overrides", [
    {"prompt": "     "}, {"prompt": "a" * 501}, {"seed": -1}, {"seed": 2**32},
    {"num_steps": 51}, {"num_steps": 4}, {"guidance_scale": 16},
    {"adapter_scale": 2}, {"model_choice": "unknown"}, {"unrequested_option": 1},
])
def test_invalid_requests_never_reach_model(setup, overrides):
    client, app, config, engine = setup
    response = client.post("/portraits", json={"prompt": "A cat portrait", **overrides})
    assert response.status_code == 422
    assert engine.calls == []


def test_missing_model_is_explicit_and_does_not_download(setup):
    client, app, config, engine = setup
    (config.base_model_dir / "model_index.json").unlink()
    response = client.post("/portraits", json={"prompt": "A cat portrait"})
    assert response.status_code == 503
    assert "missing" in response.json()["detail"]
    assert engine.calls == []
    assert not app.state.generation_lock.locked()


def test_missing_adapter_does_not_silently_use_base(setup):
    client, app, config, engine = setup
    (config.adapter_dir / "pytorch_lora_weights.safetensors").unlink()
    response = client.post("/portraits", json={"prompt": "A cat portrait", "model_choice": "lora"})
    assert response.status_code == 503
    assert engine.calls == []
    assert client.post("/portraits", json={"prompt": "A cat portrait", "model_choice": "base"}).status_code == 200


def test_busy_model_rejects_concurrent_job(setup):
    client, app, config, engine = setup
    app.state.generation_lock.acquire()
    try:
        response = client.post("/portraits", json={"prompt": "A cat portrait"})
        assert response.status_code == 409
        assert engine.calls == []
        assert client.get("/health").json()["portrait"]["status"] == "busy"
    finally:
        app.state.generation_lock.release()


def test_runtime_error_releases_lock_and_reports_failure(setup):
    client, app, config, engine = setup
    def fail(**kwargs):
        raise RuntimeError("test CUDA allocation failure")
    engine.generate = fail
    response = client.post("/portraits", json={"prompt": "A cat portrait"})
    assert response.status_code == 503
    assert not app.state.generation_lock.locked()
    assert "allocation failure" in client.get("/health").json()["portrait"]["last_error"]
    assert client.get("/health").json()["portrait"]["model_loaded"] is False


def test_filtered_image_is_not_returned_as_success(setup):
    client, app, config, engine = setup
    engine.generate = lambda **kwargs: (Image.new("RGB", (8, 8)), {"nsfw_content_detected": True})
    response = client.post("/portraits", json={"prompt": "A cat portrait"})
    assert response.status_code == 422
    assert "image_base64" not in response.json()
    assert not app.state.generation_lock.locked()


def test_assignment1_bigram_endpoint_preserved(setup):
    client, *_ = setup
    response = client.post("/generate", json={"start_word": "the", "length": 5})
    assert response.status_code == 200
    assert response.json()["generated_text"].split()[0] == "the"
    assert len(response.json()["generated_text"].split()) <= 5
    assert client.post("/generate", json={"start_word": "unseen", "length": 5}).status_code == 400
