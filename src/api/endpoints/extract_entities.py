from typing import Annotated

from fastapi import File, HTTPException, UploadFile

from src.api.base_router import BaseRouter
from src.core.orchestrator import extract_entities_impl
from src.models import DocumentModelResponse
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


def handle_error(exception: Exception) -> None:
    """
    Handles errors by logging and raising HTTP exceptions.

    Parameters
    ----------
    exception : Exception
        The exception to handle

    Raises
    ------
    HTTPException
        With a 500 status code and the exception detail
    """
    logger.error(f"Error: {exception}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(exception)) from exception


async def extract_entities(file: Annotated[UploadFile, File(description="JPG document to extract entities from")]):
    """
    Extract entities from uploaded JPG document.

    Parameters
    ----------
    file : UploadFile
        The uploaded JPG file to process

    Returns
    -------
    DocumentModelResponse
        The extracted entities and metadata

    Raises
    ------
    HTTPException
        If file is not JPG or processing fails
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    if not file.filename.lower().endswith((".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only JPG/JPEG files are allowed")

    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400, detail=f"Invalid content type: {file.content_type}. Only JPG/JPEG files are allowed"
        )

    try:
        content = await file.read()

        logger.info(f"Processing file: {file}")

        response = await extract_entities_impl(content)

        logger.info(response)

        return DocumentModelResponse.model_validate(response)

    except Exception as e:
        handle_error(e)


router = BaseRouter(
    path="/extract-entities", endpoint=extract_entities, tags=["extract_entities"], response_model=DocumentModelResponse
).router
