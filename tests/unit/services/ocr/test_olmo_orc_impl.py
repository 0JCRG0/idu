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
        mock_request.assert_called_once_with("test_image.png", None, None)
        mock_parse.assert_called_once_with('{"natural_text": "Extracted text"}')

    @patch.object(OlmoOCREngine, "_olmo_ocr_hf_endpoint_request")
    @patch.object(OlmoOCREngine, "_parse_ocr_response")
    def test_extract_text_from_image_with_anchor(self, mock_parse, mock_request, ocr_engine):
        """Test text extraction with anchor."""
        mock_request.return_value = '{"natural_text": "Extracted text with anchor"}'
        mock_parse.return_value = "Extracted text with anchor"

        result = ocr_engine.extract_text_from_image("test_image.png", anchor=True)

        assert result == "Extracted text with anchor"
        mock_request.assert_called_once_with("test_image.png", None, True)
        mock_parse.assert_called_once_with('{"natural_text": "Extracted text with anchor"}')

    @patch.object(OlmoOCREngine, "_olmo_ocr_hf_endpoint_request_async")
    @patch.object(OlmoOCREngine, "_parse_ocr_response")
    @pytest.mark.asyncio
    async def test_extract_text_from_image_async_success(self, mock_parse, mock_request, ocr_engine):
        """Test successful async text extraction from image."""
        mock_request.return_value = '{"natural_text": "Async extracted text"}'
        mock_parse.return_value = "Async extracted text"

        result = await ocr_engine.extract_text_from_image_async("test_image.png")

        assert result == "Async extracted text"
        mock_request.assert_called_once_with("test_image.png", None, None)
        mock_parse.assert_called_once_with('{"natural_text": "Async extracted text"}')

    @patch("src.services.ocr.olmo_ocr_impl.Image.open")
    @patch("src.services.ocr.olmo_ocr_impl.TesseractOCREngine")
    def test_prepare_image_and_prompt_with_path(self, mock_tesseract_class, mock_image_open, ocr_engine):
        """Test image preparation with file path."""
        mock_img = Mock()
        mock_image_open.return_value = mock_img
        mock_tesseract = Mock()
        mock_tesseract_class.return_value = mock_tesseract
        mock_tesseract.extract_text_from_image.return_value = "tesseract text"

        with patch("io.BytesIO") as mock_bytesio, patch("base64.b64encode", return_value=b"encoded_image") as _:
            mock_buffer = Mock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b"image_data"

            base64_img, prompt = ocr_engine._prepare_image_and_prompt("test.png", None, True)

            assert base64_img == "encoded_image"
            assert "tesseract text" in prompt
            mock_image_open.assert_called_once_with("test.png")
            mock_img.save.assert_called_once_with(mock_buffer, format="PNG")

    @patch("src.services.ocr.olmo_ocr_impl.Image.open")
    def test_prepare_image_and_prompt_with_bytes(self, mock_image_open, ocr_engine):
        """Test image preparation with bytes input."""
        mock_img = Mock()
        mock_image_open.return_value = mock_img
        image_bytes = b"fake_image_data"

        with patch("io.BytesIO") as mock_bytesio, patch("base64.b64encode", return_value=b"encoded_image") as _:
            mock_buffer = Mock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b"image_data"

            base64_img, prompt = ocr_engine._prepare_image_and_prompt(None, image_bytes, None)

            assert base64_img == "encoded_image"
            assert (
                prompt
                == "Just return the plain text representation of this document as if you were reading it naturally."
            )

    def test_prepare_image_and_prompt_invalid_inputs(self, ocr_engine):
        """Test image preparation with invalid inputs."""
        # Both path and bytes provided
        with pytest.raises(AssertionError, match="Only one of image_path or image_input should be provided"):
            ocr_engine._prepare_image_and_prompt("path.png", b"bytes", None)

        # Neither path nor bytes provided
        with pytest.raises(AssertionError, match="Invalid type for image_path or image_input"):
            ocr_engine._prepare_image_and_prompt(None, None, None)

        # Wrong types
        with pytest.raises(AssertionError, match="Invalid type for image_path or image_input"):
            ocr_engine._prepare_image_and_prompt(123, None, None)

    def test_create_chat_messages(self, ocr_engine):
        """Test chat message creation."""
        base64_img = "encoded_image_data"
        prompt = "test prompt"

        messages = ocr_engine._create_chat_messages(base64_img, prompt)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "image_url"
        assert messages[0]["content"][0]["image_url"]["url"] == "data:image/png;base64,encoded_image_data"
        assert messages[0]["content"][1]["type"] == "text"
        assert messages[0]["content"][1]["text"] == "test prompt"

    @patch("src.services.ocr.olmo_ocr_impl.AsyncOpenAI")
    @patch.object(OlmoOCREngine, "_prepare_image_and_prompt")
    @patch.object(OlmoOCREngine, "_create_chat_messages")
    @pytest.mark.asyncio
    async def test_olmo_ocr_hf_endpoint_request_async_success(
        self, mock_create_messages, mock_prepare, mock_async_openai, ocr_engine
    ):
        """Test successful async HF endpoint request."""
        mock_prepare.return_value = ("base64_image", "test prompt")
        mock_create_messages.return_value = [{"role": "user", "content": "test"}]

        mock_client = Mock()
        mock_async_openai.return_value = mock_client
        mock_completion = Mock()

        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"natural_text": "async response"}'
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_client.chat.completions.create = mock_create

        result = await ocr_engine._olmo_ocr_hf_endpoint_request_async("test.png")

        assert result == '{"natural_text": "async response"}'
        mock_async_openai.assert_called_once_with(base_url=ocr_engine.endpoint_url, api_key=ocr_engine.api_key)

    @patch("src.services.ocr.olmo_ocr_impl.AsyncOpenAI")
    @patch.object(OlmoOCREngine, "_prepare_image_and_prompt")
    @patch.object(OlmoOCREngine, "_create_chat_messages")
    @pytest.mark.asyncio
    async def test_olmo_ocr_hf_endpoint_request_async_no_content(
        self, mock_create_messages, mock_prepare, mock_async_openai, ocr_engine
    ):
        """Test async HF endpoint request with no content."""
        mock_prepare.return_value = ("base64_image", "test prompt")
        mock_create_messages.return_value = [{"role": "user", "content": "test"}]

        mock_client = Mock()
        mock_async_openai.return_value = mock_client
        mock_completion = Mock()

        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]

        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_client.chat.completions.create = mock_create

        with pytest.raises(AssertionError, match="No text extracted from the image"):
            await ocr_engine._olmo_ocr_hf_endpoint_request_async("test.png")

    @patch("src.services.ocr.olmo_ocr_impl.OpenAI")
    @patch.object(OlmoOCREngine, "_prepare_image_and_prompt")
    @patch.object(OlmoOCREngine, "_create_chat_messages")
    def test_olmo_ocr_hf_endpoint_request_no_content(self, mock_create_messages, mock_prepare, mock_openai, ocr_engine):
        """Test HF endpoint request with no content."""
        mock_prepare.return_value = ("base64_image", "test prompt")
        mock_create_messages.return_value = [{"role": "user", "content": "test"}]

        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_completion = Mock()

        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        with pytest.raises(AssertionError, match="No text extracted from the image"):
            ocr_engine._olmo_ocr_hf_endpoint_request("test.png")
