from pydantic import BaseModel


class HuggingFaceAPIKeys(BaseModel):
    """Model representing HF API Keys."""

    access_token: str
    url: str


class APIKeys(BaseModel):
    """Model representing the API Keys of different LLMs."""

    openai: str
    anthropic: str
    hf: HuggingFaceAPIKeys


class EllVariables(BaseModel):
    """Model representing the Ell variables."""

    store_path: str

class DjangoSecrets(BaseModel):
    """Model representing the Django Secrets."""

    secret_key: str


class EnvVariables(BaseModel):
    """Model representing all the environment variables."""

    api_keys: APIKeys
    django_secrets: DjangoSecrets
    ell: EllVariables

