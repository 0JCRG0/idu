from typing import Any

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class HealthCheck(BaseModel):
    """Base model for the health check."""

    status: str = "ok"

class SuccessfulResponse(BaseModel):
    """Base model for a successful response."""

    status: str
    message: Any

class APIKeys(BaseModel):
    """Model representing the API Keys of different LLMs."""

    openai: str
    anthropic: str

class EnvVariables(BaseModel):
    """Model representing all the environment variables."""

    api_keys: APIKeys

