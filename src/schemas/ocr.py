from pydantic import BaseModel


class OlmoOCRResponse(BaseModel):
    """Model representing the response from the Olmo OCR engine."""

    primary_language: str
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: str