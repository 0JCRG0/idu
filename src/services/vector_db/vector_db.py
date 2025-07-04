from typing import Any, Literal

from src.services.vector_db.base import VectorDBBase
from src.services.vector_db.chroma_impl import ChromaVectorDB


class VectorDBFactory:
    """Factory class for creating vector database instances."""
    
    @staticmethod
    def create(db_type: Literal["chromadb"]) -> VectorDBBase:
        """
        Create a vector database instance based on the specified type.
        
        Args:
            db_type: Type of vector database (currently only "chromadb" supported)
            
        Returns
        -------
            Instance of the requested vector database implementation
            
        Raises
        ------
            ValueError: If an unsupported database type is specified
        """
        if db_type == "chromadb":
            return ChromaVectorDB()
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")

