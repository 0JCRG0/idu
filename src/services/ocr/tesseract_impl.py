import io

import pytesseract
from PIL import Image

from src.services.ocr.base import OCREngineBase
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class TesseractOCREngine(OCREngineBase):
    """Tesseract OCR engine implementation."""

    TIMEOUT = 5

    def extract_text_from_image(
        self, image_path: str | None = None, image_input: bytes | None = None, anchor: bool | None = None
    ) -> str:
        """
        Extract text from an image using a specific OCR.

        Args:
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | None, optional): The image input as bytes.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text from the image.

        Raises
        ------
            RuntimeError: If there is an error extracting text from the image.
            Exception: If there is an unexpected error.
        """
        try:
            if image_input and image_path:
                raise AssertionError("Both image_path and image_input cannot be provided.")
            if isinstance(image_path, str) and image_input is None:
                return pytesseract.image_to_string(Image.open(image_path), timeout=self.TIMEOUT)
            elif isinstance(image_input, bytes) and image_path is None:
                return pytesseract.image_to_string(Image.open(io.BytesIO(image_input)), timeout=self.TIMEOUT)
            else:
                raise AssertionError("Invalid type for image_path or image_input.")

        except RuntimeError as e:
            logger.error(f"RuntimeError extracting text from image: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}", exc_info=True)
            raise e

    async def extract_text_from_image_async(
        self, image_path: str | None = None, image_input: bytes | None = None, anchor: bool | None = None
    ) -> str:
        """
        Extract text from an image using the HF OCR model asynchronously.

        Args:
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | None, optional): The image input as bytes.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text.
        """
        raise NotImplementedError("Async extraction is not implemented for Tesseract OCR.")
