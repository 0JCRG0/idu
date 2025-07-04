import io
import json
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.services.ocr.olmo_ocr_impl import OlmoOCREngine


@pytest.fixture
def ocr_engine():
    """Create an OlmoOCREngine instance for testing."""
    return OlmoOCREngine()


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    img = Image.new("RGB", (100, 100), color="white")
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return img_buffer


@pytest.fixture
def mock_chat_completion():
    """Create a mock chat completion response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({"natural_text": "This is extracted text from the image."})
    return mock_response


class TestOlmoOCREngine:
    """Test cases for OlmoOCREngine class."""

    @patch("src.services.ocr.olmo_ocr_impl.OpenAI")
    @patch("src.services.ocr.olmo_ocr_impl.Image.open")
    def test_olmo_ocr_hf_endpoint_request_success(
        self, mock_image_open, mock_openai_class, ocr_engine, mock_image, mock_chat_completion
    ):
        """Test successful OCR request without anchor."""
        # Setup mocks
        mock_image_open.return_value = Image.new("RGB", (100, 100), color="white")
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_chat_completion

        result = ocr_engine._olmo_ocr_hf_endpoint_request("test_image.png")

        assert result == json.dumps({"natural_text": "This is extracted text from the image."})
        mock_openai_class.assert_called_once_with(base_url=ocr_engine.endpoint_url, api_key=ocr_engine.api_key)
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.services.ocr.olmo_ocr_impl.OpenAI")
    @patch("src.services.ocr.olmo_ocr_impl.Image.open")
    @patch("src.services.ocr.tesseract_impl.TesseractOCREngine")
    def test_olmo_ocr_hf_endpoint_request_with_anchor(
        self, mock_tesseract_class, mock_image_open, mock_openai_class, ocr_engine, mock_chat_completion
    ):
        """Test OCR request with anchor text."""
        mock_image_open.return_value = Image.new("RGB", (100, 100), color="white")
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_chat_completion

        mock_tesseract = Mock()
        mock_tesseract_class.return_value = mock_tesseract
        mock_tesseract.extract_text_from_image.return_value = "Anchor text"

        result = ocr_engine._olmo_ocr_hf_endpoint_request("test_image.png", anchor=True)

        assert result == json.dumps({"natural_text": "This is extracted text from the image."})
        mock_tesseract.extract_text_from_image.assert_called_once_with("test_image.png")

    def test_parse_ocr_response_success(self, ocr_engine):
        """Test successful parsing of OCR response."""
        response_string = '{"primary_language":"en","is_rotation_valid":true,"rotation_correction":0,"is_table":false,"is_diagram":true,"natural_text":"A few of the decisions your kids may make without you."}'  # noqa: E501

        result = ocr_engine._parse_ocr_response(response_string)
        assert result == "A few of the decisions your kids may make without you."

    def test_parse_ocr_response_invalid_json(self, ocr_engine):
        """Test parsing with invalid JSON."""
        response_string = "This is not valid JSON"

        with pytest.raises(json.JSONDecodeError):
            ocr_engine._parse_ocr_response(response_string)

    def test_parse_ocr_response_missing_field(self, ocr_engine):
        """Test parsing with missing required field."""
        response_string = json.dumps({"some_other_field": "value"})

        with pytest.raises(ValueError):
            ocr_engine._parse_ocr_response(response_string)

    @patch.object(OlmoOCREngine, "_olmo_ocr_hf_endpoint_request")
    @patch.object(OlmoOCREngine, "_parse_ocr_response")
    def test_extract_text_from_image_success(self, mock_parse, mock_request, ocr_engine):
        """Test successful text extraction from image."""
        mock_request.return_value = '{"natural_text": "Extracted text"}'
        mock_parse.return_value = "Extracted text"

        result = ocr_engine.extract_text_from_image("test_image.png")

        assert result == "Extracted text"
        mock_request.assert_called_once_with("test_image.png", None)
        mock_parse.assert_called_once_with('{"natural_text": "Extracted text"}')

    @patch.object(OlmoOCREngine, "_olmo_ocr_hf_endpoint_request")
    @patch.object(OlmoOCREngine, "_parse_ocr_response")
    def test_extract_text_from_image_with_anchor(self, mock_parse, mock_request, ocr_engine):
        """Test text extraction with anchor."""
        mock_request.return_value = '{"natural_text": "Extracted text with anchor"}'
        mock_parse.return_value = "Extracted text with anchor"

        result = ocr_engine.extract_text_from_image("test_image.png", anchor=True)

        assert result == "Extracted text with anchor"
        mock_request.assert_called_once_with("test_image.png", True)
        mock_parse.assert_called_once_with('{"natural_text": "Extracted text with anchor"}')
