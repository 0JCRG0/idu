import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.utils.file_processing import (
    get_supported_content_types,
    get_supported_extensions,
    pdf_to_png_bytes,
    validate_and_convert_image,
)


class TestPdfToPngBytes:
    """Tests for pdf_to_png_bytes conversion, including success and error scenarios."""

    @patch("src.utils.file_processing.convert_from_bytes")
    def test_pdf_to_png_bytes_success(self, mock_convert):
        """Test successful PDF to PNG conversion returns expected bytes."""
        mock_image = MagicMock(spec=Image.Image)
        mock_convert.return_value = [mock_image]

        pdf_bytes = b"fake_pdf_content"
        expected_png_bytes = b"fake_png_content"

        mock_image_io = io.BytesIO(expected_png_bytes)
        with patch("io.BytesIO") as mock_bytesio:
            mock_bytesio.return_value = mock_image_io
            mock_image_io.getvalue = MagicMock(return_value=expected_png_bytes)

            result = pdf_to_png_bytes(pdf_bytes)

            assert result == expected_png_bytes
            mock_convert.assert_called_once_with(pdf_bytes, first_page=1, last_page=1, dpi=200)
            mock_image.save.assert_called_once_with(mock_image_io, format="PNG")

    @patch("src.utils.file_processing.convert_from_bytes")
    def test_pdf_to_png_bytes_no_images(self, mock_convert):
        """Test PDF conversion raises exception when no images are output."""
        mock_convert.return_value = []
        pdf_bytes = b"fake_pdf_content"

        with pytest.raises(Exception, match="Failed to convert PDF to PNG.*PDF conversion resulted in no images"):
            pdf_to_png_bytes(pdf_bytes)

    @patch("src.utils.file_processing.convert_from_bytes")
    def test_pdf_to_png_bytes_conversion_error(self, mock_convert):
        """Test PDF conversion error raises appropriate exception."""
        mock_convert.side_effect = Exception("PDF library error")
        pdf_bytes = b"fake_pdf_content"

        with pytest.raises(Exception, match="Failed to convert PDF to PNG.*PDF library error"):
            pdf_to_png_bytes(pdf_bytes)


class TestValidateAndConvertImage:
    """Tests for validate_and_convert_image for all supported and unsupported cases."""

    @patch("src.utils.file_processing.pdf_to_png_bytes")
    def test_validate_and_convert_pdf_by_extension(self, mock_pdf_convert):
        """Test PDF file conversion by extension."""
        expected_result = b"converted_png"
        mock_pdf_convert.return_value = expected_result

        result = validate_and_convert_image(b"pdf_content", "application/pdf", "document.pdf")

        assert result == expected_result
        mock_pdf_convert.assert_called_once_with(b"pdf_content")

    @patch("src.utils.file_processing.pdf_to_png_bytes")
    def test_validate_and_convert_pdf_by_content_type(self, mock_pdf_convert):
        """Test PDF content conversion by content type regardless of file extension."""
        expected_result = b"converted_png"
        mock_pdf_convert.return_value = expected_result

        result = validate_and_convert_image(b"pdf_content", "application/pdf", "document.txt")

        assert result == expected_result
        mock_pdf_convert.assert_called_once_with(b"pdf_content")

    def test_validate_and_convert_png_by_extension(self):
        """Test PNG file is accepted by extension."""
        content = b"png_content"
        result = validate_and_convert_image(content, "image/png", "image.png")
        assert result == content

    def test_validate_and_convert_png_by_content_type(self):
        """Test PNG file is accepted by content type."""
        content = b"png_content"
        result = validate_and_convert_image(content, "image/png", "image.txt")
        assert result == content

    def test_validate_and_convert_jpg_by_extension(self):
        """Test JPG file is accepted by extension."""
        content = b"jpg_content"
        result = validate_and_convert_image(content, "image/jpeg", "image.jpg")
        assert result == content

    def test_validate_and_convert_jpeg_by_extension(self):
        """Test JPEG file is accepted by extension."""
        content = b"jpeg_content"
        result = validate_and_convert_image(content, "image/jpeg", "image.jpeg")
        assert result == content

    def test_validate_and_convert_jpg_by_content_type(self):
        """Test JPG file is accepted by content type."""
        content = b"jpg_content"
        result = validate_and_convert_image(content, "image/jpeg", "image.txt")
        assert result == content

    def test_validate_and_convert_unsupported_extension(self):
        """Test unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format: document.txt \\(content-type: text/plain\\)"):
            validate_and_convert_image(b"content", "text/plain", "document.txt")

    def test_validate_and_convert_unsupported_content_type(self):
        """Test unsupported content type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format: document.gif \\(content-type: image/gif\\)"):
            validate_and_convert_image(b"content", "image/gif", "document.gif")

    def test_validate_and_convert_case_insensitive(self):
        """Test case-insensitive extension matching works for PDF."""
        content = b"pdf_content"
        with patch("src.utils.file_processing.pdf_to_png_bytes", return_value=b"result") as mock_pdf:
            result = validate_and_convert_image(content, "text/plain", "DOCUMENT.PDF")
            assert result == b"result"
            mock_pdf.assert_called_once_with(content)


class TestSupportedFormats:
    """Tests for retrieving supported file extensions and content types."""

    def test_get_supported_extensions(self):
        """Test correct tuple of supported file extensions is returned."""
        extensions = get_supported_extensions()
        expected = (".jpg", ".jpeg", ".png", ".pdf")
        assert extensions == expected
        assert isinstance(extensions, tuple)

    def test_get_supported_content_types(self):
        """Test correct tuple of supported content types is returned."""
        content_types = get_supported_content_types()
        expected = ("image/jpeg", "image/jpg", "image/png", "application/pdf")
        assert content_types == expected
        assert isinstance(content_types, tuple)
