import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.services.vector_db.chroma_impl import ChromaVectorDB


class TestChromaVectorDB:
    """Tests for the ChromaVectorDB class."""

    @pytest.fixture
    def chroma_db(self):
        """Fixture returning a patched ChromaVectorDB instance."""
        with (
            patch("src.services.vector_db.chroma_impl.chromadb"),
            patch("src.services.vector_db.chroma_impl.OpenAIEmbeddingFunction"),
        ):
            return ChromaVectorDB()

    def test_init(self):
        """Test object initialization and defaults."""
        with (
            patch("src.services.vector_db.chroma_impl.chromadb") as mock_chromadb,
            patch("src.services.vector_db.chroma_impl.OpenAIEmbeddingFunction") as mock_embedding,
        ):
            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_embedding_instance = MagicMock()
            mock_embedding.return_value = mock_embedding_instance

            db = ChromaVectorDB()

            assert db.client == mock_client
            assert db.collection is None
            assert db._default_embedding_function == mock_embedding_instance

    def test_apply_sigmoid(self, chroma_db):
        """Test sigmoid application to distance values."""
        distances = [0.0, 0.5, 1.0, 1.5, 2.0]
        result = chroma_db._ChromaVectorDB__apply_sigmoid(distances)

        assert len(result) == len(distances)
        assert all(0 <= score <= 1 for score in result)
        assert result[0] > result[1] > result[2] > result[3] > result[4]

        expected_sigmoid_1_0 = round(float(1 / (1 + np.exp(5 * (1.0 - 1)))), 3)
        assert result[2] == expected_sigmoid_1_0

    def test_get_or_create_collection_default_params(self, chroma_db):
        """Test collection creation with default parameters."""
        mock_collection = MagicMock()
        chroma_db.client.get_or_create_collection.return_value = mock_collection

        result = chroma_db.get_or_create_collection()

        assert result == mock_collection
        assert chroma_db.collection == mock_collection
        chroma_db.client.get_or_create_collection.assert_called_once()
        call_args = chroma_db.client.get_or_create_collection.call_args
        assert call_args[1]["name"] == "idu_collection"
        assert call_args[1]["embedding_function"] == chroma_db._default_embedding_function
        assert "description" in call_args[1]["metadata"]
        assert "created" in call_args[1]["metadata"]

    def test_get_or_create_collection_custom_params(self, chroma_db):
        """Test collection creation with custom parameters."""
        mock_collection = MagicMock()
        mock_embedding_func = MagicMock()
        custom_metadata = {"custom": "data"}
        chroma_db.client.get_or_create_collection.return_value = mock_collection

        result = chroma_db.get_or_create_collection(
            name="custom_collection", embedding_function=mock_embedding_func, metadata=custom_metadata
        )

        assert result == mock_collection
        chroma_db.client.get_or_create_collection.assert_called_once_with(
            name="custom_collection", embedding_function=mock_embedding_func, metadata=custom_metadata
        )

    def test_add_docs_with_ids(self, chroma_db):
        """Test adding documents with provided IDs."""
        chroma_db.collection = MagicMock()
        documents = ["doc1", "doc2"]
        metadatas = [{"type": "test1"}, {"type": "test2"}]
        ids = ["id1", "id2"]

        chroma_db.add_docs(documents, metadatas, ids)

        chroma_db.collection.add.assert_called_once_with(ids=ids, documents=documents, metadatas=metadatas)

    def test_add_docs_without_ids(self, chroma_db):
        """Test adding documents without provided IDs."""
        chroma_db.collection = MagicMock()
        documents = ["doc1", "doc2"]
        metadatas = [{"type": "test1"}, {"type": "test2"}]

        with patch("src.services.vector_db.chroma_impl.uuid.uuid4") as mock_uuid:
            mock_uuid.side_effect = [
                uuid.UUID("12345678-1234-5678-1234-567812345678"),
                uuid.UUID("87654321-4321-8765-4321-876543218765"),
            ]

            chroma_db.add_docs(documents, metadatas)

            expected_ids = ["12345678-1234-5678-1234-567812345678", "87654321-4321-8765-4321-876543218765"]
            chroma_db.collection.add.assert_called_once_with(ids=expected_ids, documents=documents, metadatas=metadatas)

    def test_add_docs_no_collection(self, chroma_db):
        """Test error when adding docs without initialized collection."""
        documents = ["doc1"]
        metadatas = [{"type": "test"}]

        with pytest.raises(ValueError, match="Collection not initialized"):
            chroma_db.add_docs(documents, metadatas)

    def test_add_docs_metadata_length_mismatch(self, chroma_db):
        """Test error when docs/metadatas length mismatch."""
        chroma_db.collection = MagicMock()
        documents = ["doc1", "doc2"]
        metadatas = [{"type": "test1"}]
        ids = ["id1", "id2"]

        with pytest.raises(ValueError, match="Length of metadatas must match length of documents"):
            chroma_db.add_docs(documents, metadatas, ids)

    def test_find_similar_docs_success(self, chroma_db):
        """Test finding similar documents successfully."""
        chroma_db.collection = MagicMock()
        mock_query_result = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"type": "test1"}, {"type": "test2"}]],
            "distances": [[0.2, 0.8]],
        }
        chroma_db.collection.query.return_value = mock_query_result

        result = chroma_db.find_similar_docs("test query", n_results=5)

        ids, documents, metadatas, distances, confidence = result
        assert ids == ["id1", "id2"]
        assert documents == ["doc1", "doc2"]
        assert metadatas == [{"type": "test1"}, {"type": "test2"}]
        assert distances == [0.2, 0.8]
        assert len(confidence) == 2
        assert all(isinstance(c, float) for c in confidence)

        chroma_db.collection.query.assert_called_once_with(query_texts=["test query"], n_results=5)

    def test_find_similar_docs_empty_results(self, chroma_db):
        """Test finding similar docs when empty results returned."""
        chroma_db.collection = MagicMock()
        mock_query_result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        chroma_db.collection.query.return_value = mock_query_result

        result = chroma_db.find_similar_docs("test query")

        ids, documents, metadatas, distances, confidence = result
        assert ids == []
        assert documents == []
        assert metadatas == []
        assert distances == []
        assert confidence == []

    def test_find_similar_docs_no_collection(self, chroma_db):
        """Test error when finding similar docs without collection."""
        with pytest.raises(ValueError, match="Collection not initialized"):
            chroma_db.find_similar_docs("test query")
