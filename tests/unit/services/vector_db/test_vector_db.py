import pytest

from src.services.vector_db.chroma_impl import ChromaVectorDB
from src.services.vector_db.vector_db import VectorDBFactory


class TestVectorDBFactory:
    """Test the VectorDBFactory class."""

    def test_create_chromadb(self):
        """Test creating a ChromaVectorDB instance."""
        db = VectorDBFactory.create("chromadb")
        assert isinstance(db, ChromaVectorDB)

    def test_create_unsupported_db_type(self):
        """Test creating an unsupported vector database type."""
        with pytest.raises(ValueError, match="Unsupported vector database type: unsupported"):
            VectorDBFactory.create("unsupported")  # type: ignore
