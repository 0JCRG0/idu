from abc import ABC

from src.services.vector_db.base import VectorDBBase


class TestVectorDBBase:
    """Test the VectorDBBase class."""

    def test_is_abstract_base_class(self):
        """Test that VectorDBBase is an abstract base class."""
        assert issubclass(VectorDBBase, ABC)

    def test_has_required_abstract_methods(self):
        """Test that VectorDBBase has the required abstract methods."""
        abstract_methods = VectorDBBase.__abstractmethods__
        expected_methods = {"get_or_create_collection", "add_docs", "find_similar_docs"}
        assert abstract_methods == expected_methods

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of VectorDBBase works."""

        class ConcreteVectorDB(VectorDBBase):
            def get_or_create_collection(self, name="test", embedding_function=None, metadata=None):
                return f"collection_{name}"

            def add_docs(self, documents, metadatas, ids=None):
                pass

            def find_similar_docs(self, query_text, n_results=10):
                return ([], [], [], [], [])

        db = ConcreteVectorDB()
        assert db.get_or_create_collection() == "collection_test"
        assert db.get_or_create_collection("custom") == "collection_custom"
