from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.orchestrator import extract_entities_impl


class TestExtractEntitiesImpl:
    """Unit tests for the extract_entities_impl function."""

    @pytest.fixture
    def mock_image_input(self):
        """Mock image input as bytes."""
        return b"fake_image_data"

    @pytest.fixture
    def mock_ocr_response(self):
        """Mock OCR response."""
        return "This is extracted text from a document"

    @pytest.fixture
    def mock_vector_db_response(self):
        """Mock vector DB response."""
        return (["id1"], ["doc1"], [{"document_type": "invoice"}], [0.2], [0.8])

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response."""
        return MagicMock(text="invoice")

    @pytest.fixture
    def mock_extraction_response(self):
        """Mock extraction response."""
        return MagicMock(text='{"field1": "value1", "field2": "value2"}')

    @patch("src.core.orchestrator.extract_valid_json")
    @patch("src.core.orchestrator.extract_entities_from_doc")
    @patch("src.core.orchestrator.create_extraction_prompt")
    @patch("src.core.orchestrator.validate_document_type")
    @patch("src.core.orchestrator.create_document_type_validation_prompt")
    @patch("src.core.orchestrator.VectorDBFactory")
    @patch("src.core.orchestrator.OCREngineFactory")
    @patch("src.core.orchestrator.time.time")
    @pytest.mark.asyncio
    async def test_extract_entities_impl_success(
        self,
        mock_time,
        mock_ocr_factory,
        mock_vector_factory,
        mock_validation_prompt,
        mock_validate_doc_type,
        mock_extraction_prompt,
        mock_extract_entities,
        mock_extract_json,
        mock_image_input,
        mock_ocr_response,
        mock_vector_db_response,
        mock_llm_response,
        mock_extraction_response,
    ):
        """Test the extract_entities_impl function with a successful extraction."""
        mock_time.side_effect = [1000.0, 1000.0, 1005.5]

        mock_ocr = AsyncMock()
        mock_ocr.extract_text_from_image_async.return_value = mock_ocr_response
        mock_ocr_factory.create.return_value = mock_ocr

        mock_vector_db = MagicMock()
        mock_vector_db.find_similar_docs.return_value = mock_vector_db_response
        mock_vector_factory.create.return_value = mock_vector_db

        mock_validation_prompt.return_value = "validation prompt"
        mock_validate_doc_type.return_value = mock_llm_response
        mock_extraction_prompt.return_value = "extraction prompt"
        mock_extract_entities.return_value = mock_extraction_response
        mock_extract_json.return_value = {"field1": "value1", "field2": "value2"}

        result = await extract_entities_impl(mock_image_input)

        assert result == {
            "document_type": "invoice",
            "confidence": 0.8,
            "entities": {"field1": "value1", "field2": "value2"},
            "processing_time": 0.0,
        }

        mock_ocr.extract_text_from_image_async.assert_called_once_with(image_input=mock_image_input)
        mock_vector_factory.create.assert_called_once_with("chromadb")
        mock_vector_db.get_or_create_collection.assert_called_once()
        mock_vector_db.find_similar_docs.assert_called_once_with(mock_ocr_response)

    @patch("src.core.orchestrator.OCREngineFactory")
    @pytest.mark.asyncio
    async def test_extract_entities_impl_ocr_failure(self, mock_ocr_factory, mock_image_input):
        """Test the extract_entities_impl function when OCR fails."""
        mock_ocr = AsyncMock()
        mock_ocr.extract_text_from_image_async.side_effect = Exception("OCR failed")
        mock_ocr_factory.create.return_value = mock_ocr

        with pytest.raises(Exception, match="OCR failed"):
            await extract_entities_impl(mock_image_input)

    @patch("src.core.orchestrator.validate_document_type")
    @patch("src.core.orchestrator.create_document_type_validation_prompt")
    @patch("src.core.orchestrator.VectorDBFactory")
    @patch("src.core.orchestrator.OCREngineFactory")
    @pytest.mark.asyncio
    async def test_extract_entities_impl_invalid_document_type(
        self,
        mock_ocr_factory,
        mock_vector_factory,
        mock_validation_prompt,
        mock_validate_doc_type,
        mock_image_input,
        mock_ocr_response,
        mock_vector_db_response,
    ):
        """Test the extract_entities_impl function when the document type is invalid."""
        mock_ocr = AsyncMock()
        mock_ocr.extract_text_from_image_async.return_value = mock_ocr_response
        mock_ocr_factory.create.return_value = mock_ocr

        mock_vector_db = MagicMock()
        mock_vector_db.find_similar_docs.return_value = mock_vector_db_response
        mock_vector_factory.create.return_value = mock_vector_db

        mock_validation_prompt.return_value = "validation prompt"
        mock_validate_doc_type.return_value = MagicMock(text="invalid_document_type")

        with pytest.raises(AssertionError, match="Document type validation failed"):
            await extract_entities_impl(mock_image_input)

    @patch("src.core.orchestrator.VectorDBFactory")
    @patch("src.core.orchestrator.OCREngineFactory")
    @pytest.mark.asyncio
    async def test_extract_entities_impl_vector_db_failure(
        self, mock_ocr_factory, mock_vector_factory, mock_image_input, mock_ocr_response
    ):
        """Test the extract_entities_impl function when the vector DB fails."""
        mock_ocr = AsyncMock()
        mock_ocr.extract_text_from_image_async.return_value = mock_ocr_response
        mock_ocr_factory.create.return_value = mock_ocr

        mock_vector_db = MagicMock()
        mock_vector_db.find_similar_docs.side_effect = Exception("Vector DB failed")
        mock_vector_factory.create.return_value = mock_vector_db

        with pytest.raises(Exception, match="Vector DB failed"):
            await extract_entities_impl(mock_image_input)
