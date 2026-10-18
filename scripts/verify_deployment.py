"""Exercise a running deployment; save real responses and decoded image evidence.

Unlike tests/test_api.py, this script does not inject or mock any model. The
portrait request runs the deployed engine. A failed request makes the command
fail and its report is saved with status=failed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-choice", choices=["base", "lora"], default="lora")
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "08_Evidence" / "deployment")
    parser.add_argument("--skip-portrait", action="store_true", help="Endpoint-only test, not a successful image-generation test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"url": args.url, "model_choice": args.model_choice, "num_steps": args.num_steps, "mocked": False, "portrait_requested": not args.skip_portrait, "checks": {}, "status": "running"}

    def call(path, body=None):
        request = urllib.request.Request(args.url.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            content = response.read()
            return response.status, json.loads(content) if "json" in response.headers.get("Content-Type", "") else content.decode()

    try:
        status, health = call("/health")
        assert status == 200 and health["status"] == "ok"
        report["checks"]["health_before"] = health
        status, page = call("/")
        assert status == 200 and "CatStudio" in page
        report["checks"]["frontend"] = "passed"
        status, schema = call("/openapi.json")
        assert "/portraits" in schema["paths"] and "/generate" in schema["paths"] and "/embedding" in schema["paths"]
        report["checks"]["openapi"] = "passed"
        status, text = call("/generate", {"start_word": "the", "length": 5})
        assert status == 200 and text["generated_text"].startswith("the")
        report["checks"]["assignment1_text"] = text
        status, embedding = call("/embedding?word=apple")
        assert status == 200 and embedding["dimensions"] == 300 and len(embedding["vector"]) == 300
        report["checks"]["assignment1_embedding"] = {key: value for key, value in embedding.items() if key != "vector"}
        try:
            call("/portraits", {"prompt": "A cat portrait", "seed": -1})
            raise AssertionError("Invalid negative seed was accepted")
        except urllib.error.HTTPError as exc:
            assert exc.code == 422
            report["checks"]["invalid_seed_rejected"] = exc.code
        if not args.skip_portrait:
            request = {"prompt": "A close-up portrait of an orange and white cat, looking at the camera, soft natural light, plain cream background", "seed": 42, "model_choice": args.model_choice, "num_steps": args.num_steps, "guidance_scale": 7.5, "adapter_scale": 1.0}
            started = time.perf_counter()
            status, portrait = call("/portraits", request)
            duration = time.perf_counter() - started
            assert status == 200
            content = base64.b64decode(portrait["image_base64"])
            image = Image.open(BytesIO(content))
            image.load()
            assert image.size == (512, 512), image.size
            assert portrait["metadata"]["model_choice"] == args.model_choice
            assert not portrait["metadata"].get("nsfw_content_detected", False)
            filename = f"{args.model_choice}_seed42_steps{args.num_steps}.png"
            (args.output / filename).write_bytes(content)
            report["checks"]["portrait"] = {"request": request, "metadata": portrait["metadata"], "request_seconds": round(duration, 3), "png": filename, "sha256": hashlib.sha256(content).hexdigest(), "size": list(image.size)}
        report["checks"]["health_after"] = call("/health")[1]
        report["status"] = "passed" if not args.skip_portrait else "endpoint_checks_passed_portrait_not_tested"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        filename = f"deployment_{args.model_choice}.json" if not args.skip_portrait else "deployment_endpoints_only.json"
        (args.output / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
