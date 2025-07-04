import base64
import io
import json

from openai import OpenAI
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.constants import HF_SECRETS
from src.llm.prompts import default_olmocr_prompt, prompt_olmocr_with_anchor
from src.models import OlmoOCRResponse
from src.services.ocr.base import OCREngineBase
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class OlmoOCREngine(OCREngineBase):
    """OCR Engine using the HF OCR model."""

    def __init__(self):
        self.api_key = HF_SECRETS.access_token
        self.endpoint_url = HF_SECRETS.url

    # NOTE: Retrying this many times is required due to cold starts of the HF endpoint.
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _olmo_ocr_hf_endpoint_request(self, image_path: str, anchor: bool | None = None) -> str:
        """
        Make a request to the HF endpoint for the Olmo OCR model.

        Args
        ----
            image_path (str): The path to the image to be OCRed.
            anchor (bool | None, optional): Whether to use the anchor prompt. Defaults to None.

        Return
        -------
            str: The response from the HF endpoint.

        Raise
        ------
            Exception: If the request fails.
        """
        try:
            client = OpenAI(base_url=self.endpoint_url, api_key=self.api_key)

            img = Image.open(image_path)

            png_buffer = io.BytesIO()
            img.save(png_buffer, format="PNG")
            png_buffer.seek(0)

            png_base64 = base64.b64encode(png_buffer.getvalue()).decode("utf-8")

            prompt = default_olmocr_prompt()
            if anchor is not None:
                # NOTE: Sometimes OlmoOCR performs better when the anchor text is provided.
                from src.services.ocr.tesseract_impl import TesseractOCREngine

                tessaract_extraction = TesseractOCREngine().extract_text_from_image(image_path)
                prompt = prompt_olmocr_with_anchor(tessaract_extraction)

            chat_completion = client.chat.completions.create(
                model="tgi",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                top_p=None,
                temperature=None,
                max_tokens=1000,
                stream=False,
                seed=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
            )

            content = chat_completion.choices[0].message.content
            if not content:
                raise AssertionError("No text extracted from the image.")
            return content
        except Exception as e:
            logger.error(f"Error in _olmo_ocr_hf_endpoint_request: {e}")
            raise e

    def _parse_ocr_response(self, response_string: str) -> str:
        """
        Parse and validate an OCR response string.

        Args:
            response_string: JSON string containing OCR response data

        Returns
        -------
            OlmoOCRResponse object if validation succeeds, None if it fails

        Raises
        ------
            json.JSONDecodeError: If the string is not valid JSON
            pydantic.ValidationError: If the JSON doesn't match the model schema
        """
        try:
            data = json.loads(response_string)

            response = OlmoOCRResponse.model_validate(data)

            return response.natural_text

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise e

        except ValueError as e:
            logger.error(f"Validation failed: {e}")
            raise e

    def extract_text_from_image(self, image_path: str, anchor: bool | None = None) -> str:
        """
        Extract text from an image using the HF OCR model.

        Args:
            image_path (str): The path to the image file.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text.
        """
        result = self._olmo_ocr_hf_endpoint_request(image_path, anchor)
        validated_result = self._parse_ocr_response(result)
        return validated_result
