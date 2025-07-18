from typing import Any

from pydantic import BaseModel


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

class DjangoSecrets(BaseModel):
    """Model representing the Django Secrets."""

    secret_key: str


class EnvVariables(BaseModel):
    """Model representing all the environment variables."""

    api_keys: APIKeys
    django_secrets: DjangoSecrets


class OlmoOCRResponse(BaseModel):
    """Model representing the response from the Olmo OCR engine."""

    primary_language: str
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: str


class DocumentModelResponse(BaseModel):
    """Response model for the document extraction endpoint."""

    document_type: str
    confidence: float | None
    entities: dict
    processing_time: float
