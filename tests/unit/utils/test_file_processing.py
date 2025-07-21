from unittest.mock import MagicMock, patch

import pytest

from src.utils.file_processing import (
    get_supported_content_types,
    get_supported_extensions,
    pdf_to_png_base64,
    validate_and_convert_image,
)


class TestValidateAndConvertImage:
    """Tests for validate_and_convert_image for all supported and unsupported cases."""

    @patch("src.utils.file_processing.pdf_to_png_base64")
    def test_validate_and_convert_pdf_by_extension(self, mock_pdf_convert):
        """Test PDF file conversion by extension."""
        expected_result = b"converted_png"
        mock_pdf_convert.return_value = expected_result

        result = validate_and_convert_image(b"pdf_content", "application/pdf", "document.pdf")

        assert result == expected_result
        mock_pdf_convert.assert_called_once_with(b"pdf_content")

    @patch("src.utils.file_processing.pdf_to_png_base64")
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
        with patch("src.utils.file_processing.pdf_to_png_base64", return_value=b"result") as mock_pdf:
            result = validate_and_convert_image(content, "text/plain", "DOCUMENT.PDF")
            assert result == b"result"
            mock_pdf.assert_called_once_with(content)


class TestPdfToPngBase64:
    """Tests for pdf_to_png_base64 function."""

    @patch("src.utils.file_processing.render_pdf_to_base64png")
    @patch("src.utils.file_processing.NamedTemporaryFile")
    def test_pdf_to_png_base64_success(self, mock_temp_file, mock_render):
        """Test successful PDF to PNG base64 conversion."""
        # Setup
        mock_file = MagicMock()
        mock_temp_file.return_value.__enter__.return_value = mock_file
        mock_render.return_value = "base64_png_string"
        pdf_content = b"fake_pdf_content"

        # Execute
        result = pdf_to_png_base64(pdf_content)

        # Verify
        assert result == "base64_png_string"
        mock_file.write.assert_called_once_with(pdf_content)
        mock_render.assert_called_once_with(mock_file.name, 1, 1024)

    @patch("src.utils.file_processing.render_pdf_to_base64png")
    @patch("src.utils.file_processing.NamedTemporaryFile")
    def test_pdf_to_png_base64_render_failure(self, mock_temp_file, mock_render):
        """Test PDF conversion failure when render_pdf_to_base64png raises exception."""
        # Setup
        mock_file = MagicMock()
        mock_temp_file.return_value.__enter__.return_value = mock_file
        mock_render.side_effect = Exception("Render failed")
        pdf_content = b"fake_pdf_content"

        # Execute and verify
        with pytest.raises(Exception, match="Failed to convert PDF to PNG: Render failed"):
            pdf_to_png_base64(pdf_content)

    @patch("src.utils.file_processing.render_pdf_to_base64png")
    @patch("src.utils.file_processing.NamedTemporaryFile")
    def test_pdf_to_png_base64_temp_file_failure(self, mock_temp_file, mock_render):
        """Test PDF conversion failure when temporary file creation fails."""
        # Setup
        mock_temp_file.side_effect = Exception("Temp file creation failed")
        pdf_content = b"fake_pdf_content"

        # Execute and verify
        with pytest.raises(Exception, match="Failed to convert PDF to PNG: Temp file creation failed"):
            pdf_to_png_base64(pdf_content)


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
