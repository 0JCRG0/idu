from fastapi import FastAPI

from src.api import health
from src.api.endpoints import extract_entities
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)

app = FastAPI(
    title="Intelligent Document Understanding API",
    summary="API that extracts structured information from unstructured documents.",
    description="This API combines OCR technology, vector database retrieval, and LLMs to understand any document.",
)

app.include_router(health.router)
app.include_router(extract_entities.router)
