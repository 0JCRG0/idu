import uuid
from datetime import datetime
from typing import Any

import chromadb
import numpy as np
from chromadb import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.constants import EMBEDDING_DEFAULT_MODEL, OPENAI_API_KEY
from src.services.vector_db.base import VectorDBBase


class ChromaVectorDB(VectorDBBase):
    """ChromaDB implementation of the VectorDBBase interface."""

    def __init__(self):
        self.client = chromadb.PersistentClient()
        self.collection = None
        self._default_embedding_function = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY, model_name=EMBEDDING_DEFAULT_MODEL
        )

    def __apply_sigmoid(self, distances: list[float]) -> list[float]:
        """
        Apply sigmoid transformation to normalize distances into confidence scores.

        Transforms cosine distances into confidence scores using a sigmoid function
        with midpoint at 1.0 and steepness factor of 5. This creates a smooth
        transition from high similarity (near 0) to low similarity (near 1).

        Args:
            distances: List of cosine distances (0-2 range)

        Returns
        -------
            List of similarity scores between 0 and 1, rounded to 3 decimal places

        Example:
            >>> __apply_sigmoid([0.5, 1.0, 1.5])
            [0.993, 0.5, 0.007]  # High, medium, low similarity
        """
        return [round(float(1 / (1 + np.exp(5 * (d - 1)))), 3) for d in distances]

    def get_or_create_collection(
        self,
        name: str = "idu_collection",
        embedding_function: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Collection:
        """
        Create or get a ChromaDB collection.

        Args:
            name: Name of the collection (default: "idu_collection")
            embedding_function: Embedding function (default: OpenAI text-embedding-3-small)
            metadata: Collection metadata (default: includes description and creation time)

        Returns
        -------
            ChromaDB collection object
        """
        if embedding_function is None:
            embedding_function = self._default_embedding_function

        if metadata is None:
            metadata = {"description": "Collection for IDU API", "created": str(datetime.now())}

        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function,
            metadata=metadata,
        )

        return self.collection

    def add_docs(self, documents: list[str], metadatas: list[dict[str, str]], ids: list[str] | None = None) -> None:
        """
        Add documents to the ChromaDB collection.

        Args:
            documents: list of text documents to add
            metadatas: list of metadata dicts with document_type field
            ids: list of unique identifiers (auto-generated UUIDs if not provided)
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")

        # Generate UUIDs if ids not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Ensure metadatas list matches documents length
        elif len(metadatas) != len(documents):
            raise ValueError("Length of metadatas must match length of documents")

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,  # type: ignore
        )

    def find_similar_docs(
        self, query_text: str, n_results: int = 10
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[float], list[float]]:
        """
        Find similar documents in the collection.

        Args:
            query_text: Text to search for similar documents
            n_results: Number of results to return (default: 10)

        Returns
        -------
            tuple of (ids, documents, metadatas, distances, confidence)
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call create_collection() first.")

        query_result = self.collection.query(query_texts=[query_text], n_results=n_results)
        # Extract fields from query result
        # ChromaDB returns lists of lists, so we flatten them
        ids = query_result["ids"][0] if query_result["ids"] else []
        documents = query_result["documents"][0] if query_result["documents"] else []
        metadatas = query_result["metadatas"][0] if query_result["metadatas"] else []
        distances = query_result["distances"][0] if query_result["distances"] else []

        confidence = self.__apply_sigmoid(distances)

        return ids, documents, metadatas, distances, confidence  # type: ignore
