"""FastAPI application extending the student's Assignment 1 endpoints.

Weights load only when a portrait is requested. A health check never starts a
model download or claims that untested inference is ready. Run one Uvicorn
worker: the process-local lock serializes access to the shared diffusion model.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
import logging
import os
from pathlib import Path
import threading
import time
from typing import Annotated, Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .legacy.bigram_model import BigramModel
from .schemas import (
    EmbeddingResponse, PortraitRequest, PortraitResponse,
    TextGenerationRequest, TextGenerationResponse,
)

LOGGER = logging.getLogger("catstudio")
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CORPUS = ["The cat reads books", "The dog reads notes", "The cat writes notes"]
BASE_REQUIRED_FILES = (
    "model_index.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors",
    "safety_checker/model.safetensors", "scheduler/scheduler_config.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt",
)


@dataclass(frozen=True)
class Settings:
    base_model_dir: Path
    adapter_dir: Path
    device: str = "auto"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_model_dir=Path(os.getenv("CATSTUDIO_BASE_MODEL", str(PROJECT_ROOT / "05_Models" / "base_sd15"))).resolve(),
            adapter_dir=Path(os.getenv("CATSTUDIO_ADAPTER", str(PROJECT_ROOT / "05_Models" / "cat_lora"))).resolve(),
            device=os.getenv("CATSTUDIO_DEVICE", "auto"),
        )


def _base_available(settings: Settings) -> bool:
    return all((settings.base_model_dir / name).is_file() for name in BASE_REQUIRED_FILES)


def _adapter_available(settings: Settings) -> bool:
    # Match the released Diffusers filename accepted by PortraitEngine.
    return (settings.adapter_dir / "pytorch_lora_weights.safetensors").is_file()


def _default_engine_factory(settings: Settings):
    from .model import PortraitEngine

    return PortraitEngine(
        base_model_dir=settings.base_model_dir,
        adapter_dir=settings.adapter_dir,
        device=settings.device,
    )


def create_app(settings: Settings | None = None, engine_factory: Callable | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    factory = engine_factory or _default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bigram = BigramModel(CORPUS)
        app.state.engine = None
        app.state.embeddings = None
        app.state.generation_lock = threading.Lock()
        app.state.embedding_lock = threading.Lock()
        app.state.last_error = None
        app.state.last_success = None
        yield
        app.state.engine = None
        app.state.embeddings = None

    app = FastAPI(
        title="CatStudio", version="1.0.0", lifespan=lifespan,
        description=("Local text-to-cat diffusion with a base/LoRA comparison. "
                     "Extends Assignment 1: /embedding and /generate remain available. "
                     "Cat portraits use /portraits. All model files stay on this machine."),
    )
    app.mount("/static", StaticFiles(directory=PACKAGE_ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PACKAGE_ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health", tags=["Status"])
    def health(request: Request):
        state = request.app.state
        base_available = _base_available(config)
        adapter_available = _adapter_available(config)
        busy = state.generation_lock.locked()
        engine_status = state.engine.status() if state.engine is not None and hasattr(state.engine, "status") else {}
        loaded = bool(engine_status.get("loaded", False))
        return {
            "status": "ok",
            "service": "CatStudio",
            "portrait": {
                "status": "busy" if busy else ("loaded" if loaded else ("available" if base_available else "unavailable")),
                "base_files_present": base_available,
                "adapter_files_present": adapter_available,
                "model_loaded": loaded,
                "device_requested": config.device,
                "device_loaded": str(engine_status.get("device", "not loaded")) if loaded else "not loaded",
                "last_success": state.last_success,
                "last_error": state.last_error,
                "note": "Files present is not proof of successful inference. First generation loads and checks the model.",
            },
            "assignment1": {
                "text_generation": "available",
                "embedding": "loaded" if state.embeddings is not None else "lazy; requires en_core_web_lg 3.8.0",
            },
        }

    @app.post("/portraits", response_model=PortraitResponse, tags=["Cat portraits"])
    def portrait(body: PortraitRequest, request: Request):
        state = request.app.state
        if not state.generation_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A portrait is already being generated. Wait for it to finish.")
        try:
            if not _base_available(config):
                raise HTTPException(status_code=503, detail="Base model files are missing. Follow README model setup; no image was generated.")
            if body.model_choice == "lora" and not _adapter_available(config):
                raise HTTPException(status_code=503, detail="The trained LoRA adapter is missing. Install the trained adapter or select Base model.")
            if state.engine is None:
                state.engine = factory(config)
            started = time.perf_counter()
            image, metadata = state.engine.generate(**body.model_dump())
            if metadata.get("nsfw_content_detected"):
                raise HTTPException(status_code=422, detail="The model safety checker filtered this output. Change the description and try again.")
            output = BytesIO()
            image.save(output, format="PNG")
            metadata = dict(metadata)
            metadata["api_elapsed_seconds"] = round(time.perf_counter() - started, 3)
            state.last_error = None
            state.last_success = {"model_choice": body.model_choice, "seed": body.seed, "api_elapsed_seconds": metadata["api_elapsed_seconds"]}
            return {"image_base64": base64.b64encode(output.getvalue()).decode("ascii"), "metadata": metadata}
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("Portrait inference failed")
            state.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            raise HTTPException(status_code=503, detail="Model execution failed; no image was generated. Inspect /health and the local server log.") from exc
        finally:
            state.generation_lock.release()

    @app.post("/generate", response_model=TextGenerationResponse, tags=["Assignment 1: text generation"])
    def generate_text(body: TextGenerationRequest, request: Request):
        try:
            return {"generated_text": request.app.state.bigram.generate_text(body.start_word, body.length)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/embedding", response_model=EmbeddingResponse, tags=["Assignment 1: word embeddings"])
    def embedding(request: Request, word: Annotated[str, Query(min_length=1, max_length=100)]):
        try:
            from .legacy.embeddings import InvalidWordError, UnknownWordError, WordEmbeddingService
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Install the Assignment 1 embedding dependencies listed in requirements-legacy.txt.") from exc
        state = request.app.state
        with state.embedding_lock:
            if state.embeddings is None:
                try:
                    state.embeddings = WordEmbeddingService()
                except (OSError, RuntimeError) as exc:
                    raise HTTPException(status_code=503, detail="Install en_core_web_lg 3.8.0 using requirements-legacy.txt; no substitute vectors are returned.") from exc
        try:
            return state.embeddings.embed(word)
        except InvalidWordError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except UnknownWordError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
