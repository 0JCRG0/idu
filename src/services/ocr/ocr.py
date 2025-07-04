from typing import Literal

from src.services.ocr.base import OCREngineBase
from src.services.ocr.olmo_ocr_impl import OlmoOCREngine
from src.services.ocr.tesseract_impl import TesseractOCREngine
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class OCREngine(OCREngineBase):
    """OCR engine that uses a specific OCR engine to extract text from images."""

    def __init__(self, engine: Literal["tesseract", "olmo_ocr"] = "olmo_ocr") -> None:
        self.engine = engine

    def extract_text_from_image(self, image_path: str, anchor: bool | None = None) -> str:
        """
        Extract text from an image using a specific OCR engine.

        Args:
            image_path (str): The path to the image file.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text from the image.
        """
        if self.engine == "tesseract":
            logger.info(f"Using '{self.engine}' to extract text from image: {image_path}")
            return TesseractOCREngine().extract_text_from_image(image_path)
        elif self.engine == "olmo_ocr":
            logger.info(f"Using '{self.engine}' to extract text from image: {image_path}")
            return OlmoOCREngine().extract_text_from_image(image_path, anchor)
        else:
            raise AssertionError(f"Invalid engine: {self.engine}")
