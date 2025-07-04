import pytesseract
from PIL import Image

from src.services.ocr.base import OCREngineBase
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class TesseractOCREngine(OCREngineBase):
    """Tesseract OCR engine implementation."""

    TIMEOUT = 5

    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image using Tesseract OCR.

        Args:
            image_path (str): The path to the image file.

        Returns
        -------
            str: The extracted text from the image.

        Raises
        ------
            RuntimeError: If there is an error extracting text from the image.
            Exception: If there is an unexpected error.
        """
        try:
            return pytesseract.image_to_string(Image.open(image_path), timeout=self.TIMEOUT)
        except RuntimeError as e:
            logger.error(f"RuntimeError extracting text from image: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}", exc_info=True)
            raise e
