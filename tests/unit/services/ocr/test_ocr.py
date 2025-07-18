from src.services.ocr.ocr import OCREngineFactory


class TestOCREngine:
    """Unit tests for the OCREngine class."""

    def test_extract_text_with_tesseract(self):
        """Test text extraction using tesseract engine."""
        engine = OCREngineFactory.create("tesseract")
        result = engine.extract_text_from_image("data/docs-sm/advertisement/0000126151.jpg")
        assert result == ""

        result = engine.extract_text_from_image("data/docs-sm/advertisement/0000225081.jpg")
        assert (
            result
            == "Ls atest US. Government figures\n\n~PALLMATL GOLD 1008 -\nlower in tar’\n\nbests\nae\n\nfaeses029\n\nrt’\ntime_JUL 3\n\n"  # noqa: E501
        )
