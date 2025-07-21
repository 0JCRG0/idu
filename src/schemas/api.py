from pydantic import BaseModel


class DocumentModelResponse(BaseModel):
    """Response model for the document extraction endpoint."""

    document_type: str
    confidence: float | None
    entities: dict
    processing_time: float
