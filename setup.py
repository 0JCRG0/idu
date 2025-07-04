import asyncio
import os
import shutil

import kagglehub

from src.services.ocr.ocr import OCREngineFactory
from src.services.vector_db.vector_db import VectorDBFactory
from src.utils.dataset_preparation import read_dataset_files, split_dataset
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


def download_dataset() -> bool:
    """Downloads and moves the dataset."""
    try:
        logger.info("Starting dataset download...")
        path = kagglehub.dataset_download("shaz13/real-world-documents-collections")

        if not path or not os.path.exists(path):
            logger.error("Dataset download failed - path does not exist")
            return False

        # Move the downloaded folder to 'data' directory
        destination = "data"
        if os.path.exists(destination):
            # Remove if exists
            shutil.rmtree(destination)
        shutil.move(path, destination)

        logger.info(f"Successfully moved dataset to: {destination}")
        return True

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return False


async def populate_vector_db() -> bool:
    """Populates the vector database with document embeddings asynchronously."""
    try:
        logger.info("Starting vector database population...")

        _, _ = split_dataset()
        file_paths, metadata = read_dataset_files()

        logger.info(f"Found {len(file_paths)} files to process")

        if not file_paths:
            raise AssertionError("No files found to process")

        vector_db = VectorDBFactory.create("chromadb")
        vector_db.get_or_create_collection()

        docs = []
        successful_metadata = []
        ocr_engine = OCREngineFactory.create("olmo_ocr")

        # Process files concurrently with controlled concurrency
        batch_size = 5  # Process 5 files at a time to avoid overwhelming the API

        for i in range(0, len(file_paths), batch_size):
            batch_paths = file_paths[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]
            batch_tasks = []

            for file_path in batch_paths:
                task = ocr_engine.extract_text_from_image_async(file_path)
                batch_tasks.append(task)

            # Wait for batch to complete
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to process file {batch_paths[j]}: {str(result)}")
                    else:
                        docs.append(result)
                        successful_metadata.append(batch_metadata[j])

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")

        if not docs:
            logger.error("No documents were successfully processed")
            return False

        if len(docs) != len(successful_metadata):
            raise AssertionError(
                f"Mismatch between number of docs ({len(docs)}) and metadata ({len(successful_metadata)})"
            )

        vector_db.add_docs(docs, successful_metadata)
        logger.info(f"Successfully added {len(docs)} documents to vector database")
        return True

    except Exception as e:
        logger.error(f"Error populating vector database: {str(e)}")
        raise e


async def main():
    """Main async function to run the setup."""
    success = download_dataset()
    if success:
        success = await populate_vector_db()
        if success:
            logger.info("Setup completed successfully!")
        else:
            logger.error("Failed to populate vector database")
    else:
        logger.error("Failed to download dataset")


if __name__ == "__main__":
    asyncio.run(main())
