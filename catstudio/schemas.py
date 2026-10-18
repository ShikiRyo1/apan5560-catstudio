"""API contracts; all inference controls have finite, validated bounds."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PortraitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=5, max_length=500)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    num_steps: int = Field(default=20, ge=5, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0, allow_inf_nan=False)
    adapter_scale: float = Field(default=1.0, ge=0.0, le=1.5, allow_inf_nan=False)
    model_choice: Literal["base", "lora"] = "lora"

    @field_validator("prompt", mode="before")
    @classmethod
    def trim_prompt(cls, value):
        return value.strip() if isinstance(value, str) else value


class PortraitResponse(BaseModel):
    image_base64: str = Field(description="PNG bytes encoded as base64")
    media_type: Literal["image/png"] = "image/png"
    metadata: dict


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, max_length=100, examples=["the"])
    length: int = Field(ge=1, le=50, examples=[5])


class TextGenerationResponse(BaseModel):
    generated_text: str


class EmbeddingResponse(BaseModel):
    word: str
    normalized_word: str
    model: str
    model_version: str
    dimensions: int
    vector: list[float]
