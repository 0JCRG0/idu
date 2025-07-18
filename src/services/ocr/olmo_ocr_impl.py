import base64
import io
import json

from openai import APIStatusError, AsyncOpenAI, OpenAI
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from src.constants import HF_SECRETS
from src.llm.prompts import default_olmocr_prompt, prompt_olmocr_with_anchor
from src.models import OlmoOCRResponse
from src.services.ocr.base import OCREngineBase
from src.services.ocr.tesseract_impl import TesseractOCREngine
from src.utils.logging_helper import get_custom_logger, log_attempt_retry

logger = get_custom_logger(__name__)


class OlmoOCREngine(OCREngineBase):
    """OCR Engine using the HF OCR model."""

    def __init__(self):
        self.api_key = HF_SECRETS.access_token
        self.endpoint_url = HF_SECRETS.url

    def _prepare_image_and_prompt(
        self, image_path: str | None, image_input: bytes | None, anchor: bool | None
    ) -> tuple[str, str]:
        """
        Prepare the image and prompt for OCR processing.

        Args
        ----
            image_path (str | None): The path to the image file.
            image_input (bytes | None): The image input as bytes.
            anchor (bool | None): Whether to use an anchor for the OCR engine.

        Return
        -------
            tuple[str, str]: The base64 encoded PNG image and the prompt.

        Raise
        ------
            AssertionError: If invalid inputs are provided.
        """
        if image_path and image_input:
            raise AssertionError("Only one of image_path or image_input should be provided.")

        if isinstance(image_path, str) and image_input is None:
            img = Image.open(image_path)
        elif isinstance(image_input, bytes) and image_path is None:
            img = Image.open(io.BytesIO(image_input))
        else:
            raise AssertionError("Invalid type for image_path or image_input.")

        png_buffer = io.BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        png_base64 = base64.b64encode(png_buffer.getvalue()).decode("utf-8")

        prompt = default_olmocr_prompt()
        if anchor is not None:
            # NOTE: Sometimes OlmoOCR performs better when the anchor text is provided.

            tessaract_extraction = TesseractOCREngine().extract_text_from_image(image_path, image_input)
            prompt = prompt_olmocr_with_anchor(tessaract_extraction)

        return png_base64, prompt

    def _create_chat_messages(self, png_base64: str, prompt: str) -> list[dict]:
        """
        Create the chat messages for the OCR request.

        Args
        ----
            png_base64 (str): The base64 encoded PNG image.
            prompt (str): The OCR prompt.

        Return
        -------
            list[dict]: The chat messages.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # NOTE: Retrying this many times is required due to cold starts of the HF endpoint.
    @retry(
        stop=stop_after_attempt(7),
        wait=wait_fixed(180),
        retry=retry_if_exception_type(APIStatusError),
        after=log_attempt_retry,
    )
    def _olmo_ocr_hf_endpoint_request(
        self, image_path: str | None, image_input: bytes | None = None, anchor: bool | None = None
    ) -> str:
        """
        Make a request to the HF endpoint for the Olmo OCR model.

        Args
        ----
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | None, optional): The image input as bytes.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Return
        -------
            str: The response from the HF endpoint.

        Raise
        ------
            Exception: If the request fails.
        """
        try:
            client = OpenAI(base_url=self.endpoint_url, api_key=self.api_key)

            png_base64, prompt = self._prepare_image_and_prompt(image_path, image_input, anchor)
            messages = self._create_chat_messages(png_base64, prompt)

            chat_completion = client.chat.completions.create(
                model="tgi",
                messages=messages,  # type: ignore
                top_p=None,
                temperature=None,
                max_tokens=1000,
                stream=False,
                seed=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
            )  # type: ignore

            content = chat_completion.choices[0].message.content
            logger.info(content)
            if not content:
                raise AssertionError("No text extracted from the image.")
            return content
        except APIStatusError as e:
            logger.warning("Service unavailable. Retrying in 180 seconds...")
            raise e
        except Exception as e:
            logger.error(f"Error in _olmo_ocr_hf_endpoint_request: {e}")
            raise e

    @retry(
        stop=stop_after_attempt(7),
        wait=wait_fixed(180),
        retry=retry_if_exception_type(APIStatusError),
        after=log_attempt_retry,
    )
    async def _olmo_ocr_hf_endpoint_request_async(
        self, image_path: str | None = None, image_input: bytes | None = None, anchor: bool | None = None
    ) -> str:
        """
        Make an async request to the HF endpoint for the Olmo OCR model.

        Args
        ----
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | None, optional): The image input as bytes.
            anchor (bool | None, optional): Whether to use the anchor prompt. Defaults to None.

        Return
        -------
            str: The response from the HF endpoint.

        Raise
        ------
            Exception: If the request fails.
        """
        try:
            client = AsyncOpenAI(base_url=self.endpoint_url, api_key=self.api_key)

            png_base64, prompt = self._prepare_image_and_prompt(image_path, image_input, anchor)
            messages = self._create_chat_messages(png_base64, prompt)

            chat_completion = await client.chat.completions.create(
                model="tgi",
                messages=messages,  # type: ignore
                top_p=None,
                temperature=None,
                max_tokens=1000,
                stream=False,
                seed=None,
                stop=None,
                frequency_penalty=None,
                presence_penalty=None,
            )  # type: ignore

            content = chat_completion.choices[0].message.content
            if not content:
                raise AssertionError("No text extracted from the image.")
            return content
        except APIStatusError as e:
            logger.warning("Service unavailable, retrying in 180 seconds...")
            raise e
        except Exception as e:
            logger.error(f"Error in _olmo_ocr_hf_endpoint_request_async: {e}")
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

    @retry(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        after=log_attempt_retry,
    )
    def extract_text_from_image(
        self, image_path: str | None = None, image_input: bytes | None = None, anchor: bool | None = None
    ) -> str:
        """
        Extract text from an image using the HF OCR model.

        Args:
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | None, optional): The image input as bytes.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text.
        """
        result = self._olmo_ocr_hf_endpoint_request(image_path, image_input, anchor)
        validated_result = self._parse_ocr_response(result)
        return validated_result

    @retry(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        after=log_attempt_retry,
    )
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
        result = await self._olmo_ocr_hf_endpoint_request_async(image_path, image_input, anchor)
        validated_result = self._parse_ocr_response(result)
        return validated_result
