from unittest.mock import patch

import pytest

from src.services.ocr.tesseract_impl import TesseractOCREngine


class TestTesseractOCREngine:
    """Unit tests for the TesseractOCREngine class."""

    @pytest.fixture
    def engine(self):
        """Fixture to create an instance of TesseractOCREngine."""
        return TesseractOCREngine()

    def test_extract_text_from_image_success(self, engine):
        """Test successful text extraction."""
        result = engine.extract_text_from_image("data/1/docs-sm/advertisement/00005259.jpg")

        assert result == ": Afe-v of the decisions =\n- your kids may make without you.\n\nTIMN 0053896\n"

    @patch("src.services.ocr.tesseract_impl.pytesseract.image_to_string")
    @patch("src.services.ocr.tesseract_impl.Image.open")
    def test_extract_text_runtime_error(self, mock_image_open, mock_pytesseract, engine):
        """Test RuntimeError handling."""
        mock_pytesseract.side_effect = RuntimeError("OCR failed")

        with pytest.raises(RuntimeError, match="OCR failed"):
            engine.extract_text_from_image("/path/to/image.jpg")

    @patch("src.services.ocr.tesseract_impl.pytesseract.image_to_string")
    @patch("src.services.ocr.tesseract_impl.Image.open")
    def test_extract_text_general_exception(self, mock_image_open, mock_pytesseract, engine):
        """Test general exception handling."""
        mock_pytesseract.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            engine.extract_text_from_image("/path/to/image.jpg")
