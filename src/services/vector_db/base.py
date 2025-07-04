from abc import ABC, abstractmethod
from typing import Any

from chromadb import Collection


class VectorDBBase(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    def get_or_create_collection(
        self, 
        name: str = "idu_collection",
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None
    ) -> Collection | Any:
        """
        Create or get a collection in the vector database.
        
        Args:
            name: Name of the collection (default: "idu_collection")
            embedding_function: Function to generate embeddings (implementation specific)
            metadata: Additional metadata for the collection
            
        Returns
        -------
            Collection object (implementation specific)
        """
        pass

    @abstractmethod
    def add_docs(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]] | None = None,
        ids: list[str] | None = None
    ) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: list of text documents to add
            metadatas: list of metadata dicts (e.g., {"document_type": "folder_name"})
            ids: list of unique identifiers (auto-generated if not provided)
        """
        pass

    @abstractmethod
    def find_similar_docs(
        self,
        query_text: str,
        n_results: int = 10
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[float], list[float]]:
        """
        Find similar documents based on a query.
        
        Args:
            query_text: Text to search for similar documents
            n_results: Number of results to return (default: 10)
            
        Returns
        -------
            tuple of (ids, documents, metadatas, distances, confidence)
        """
        pass