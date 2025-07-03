from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.api.base_router import BaseRouter
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)

class Entities(BaseModel):
    """Entities extracted from the document."""

    invoice_number: str = Field(..., alias="invoice\n_\nnumber")
    date: str
    total_amount: str = Field(..., alias="total\n_\namount")
    vendor_name: str = Field(..., alias="vendor\n_\nname")


class DocumentModelResponse(BaseModel):
    """Response model for the document extraction endpoint."""

    document_type: str = Field(..., alias="document\n_\ntype")
    confidence: float
    entities: Entities
    processing_time: str = Field(..., alias="processing_\ntime")



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

def extract_entities():
    # XXX: Implement the extraction logic here
    pass


router = BaseRouter(
    path="/extract-entities",
    endpoint=extract_entities,
    tags=["extraction", "ocr", "llm", "vector-database"],
).router