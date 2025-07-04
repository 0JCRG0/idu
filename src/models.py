from typing import Any

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class HealthCheck(BaseModel):
    """Base model for the health check."""

    status: str = "ok"


class SuccessfulResponse(BaseModel):
    """Base model for a successful response."""

    status: str
    message: Any


class HuggingFaceAPIKeys(BaseModel):
    """Model representing HF API Keys."""

    access_token: str
    url: str


class APIKeys(BaseModel):
    """Model representing the API Keys of different LLMs."""

    openai: str
    anthropic: str
    hf: HuggingFaceAPIKeys


class EnvVariables(BaseModel):
    """Model representing all the environment variables."""

    api_keys: APIKeys


class OlmoOCRResponse(BaseModel):
    """Model representing the response from the Olmo OCR engine."""

    primary_language: str
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: str
