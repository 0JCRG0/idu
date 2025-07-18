import asyncio
import os
import random
import shutil
from pathlib import Path

import kagglehub
from django.core.management.base import BaseCommand, CommandError

from src.constants import ROOT_DIR
from src.services.ocr.ocr import OCREngineFactory
from src.services.vector_db.vector_db import VectorDBFactory
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class Command(BaseCommand):
    """Django management command to populate vector database with document embeddings."""

    help = "Downloads dataset and populates vector database with document embeddings"

    def add_arguments(self, parser):
        """Add custom arguments for the command."""
        parser.add_argument(
            "--dataset-path",
            type=str,
            help="Path to existing dataset (if not provided, will download from Kaggle)",
        )
        parser.add_argument(
            "--skip-download",
            action="store_true",
            help="Skip dataset download, use existing data",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,
            help="Batch size for processing files (default: 10)",
        )
        parser.add_argument(
            "--ocr-engine",
            type=str,
            default="olmo_ocr",
            choices=["tesseract", "olmo_ocr"],
            help="OCR engine to use (default: olmo_ocr)",
        )

    def handle(self, *args, **options):
        """Main command handler."""
        try:
            if not options["skip_download"]:
                self.stdout.write("Downloading dataset...")
                if not self.download_dataset():
                    raise CommandError("Failed to download dataset")
                self.stdout.write(self.style.SUCCESS("Dataset downloaded successfully"))

            self.stdout.write("Populating vector database...")
            success = asyncio.run(self.populate_vector_db(options))
            if success:
                self.stdout.write(self.style.SUCCESS("Vector database populated successfully"))
            else:
                raise CommandError("Failed to populate vector database")

        except Exception as e:
            logger.error(f"Command failed: {str(e)}")
            raise CommandError(f"Command failed: {str(e)}")

    def download_dataset(self) -> bool:
        """Downloads and moves the dataset."""
        try:
            logger.info("Starting dataset download...")
            path = kagglehub.dataset_download("shaz13/real-world-documents-collections")

            if not path or not os.path.exists(path):
                logger.error("Dataset download failed - path does not exist")
                return False

            destination = "data"
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.move(path, destination)

            logger.info(f"Successfully moved dataset to: {destination}")
            return True

        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            return False

    def split_dataset_single_test(self) -> tuple[str, str]:
        """Split dataset keeping only one file per subfolder for test rest goes to train."""
        random.seed(42)
        
        source_folder = ROOT_DIR.parent / "data" / "docs-sm"
        output_folder = ROOT_DIR.parent / "data"
        train_path = output_folder / "train"
        test_path = output_folder / "test"

        for path in [train_path, test_path]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)

        # Process each subfolder in the dataset
        for subfolder in source_folder.iterdir():
            if not subfolder.is_dir():
                continue

            # Create corresponding subfolders in train and test
            train_subfolder = train_path / subfolder.name
            test_subfolder = test_path / subfolder.name
            train_subfolder.mkdir(exist_ok=True)
            test_subfolder.mkdir(exist_ok=True)

            # Get all files in the subfolder
            files = [f for f in subfolder.iterdir() if f.is_file()]
            
            if not files:
                logger.warning(f"No files found in {subfolder.name}")
                continue

            random.shuffle(files)

            # Take only one file for test, rest for train
            test_file = files[0]
            train_files = files[1:]

            shutil.copy2(test_file, test_subfolder / test_file.name)
            
            for file in train_files:
                shutil.copy2(file, train_subfolder / file.name)

            logger.info(f"Processed {subfolder.name}: {len(train_files)} train, 1 test")

        logger.info("Dataset split complete!")
        logger.info(f"Train set: {train_path.name}")
        logger.info(f"Test set: {test_path.name}")

        return train_path.name, test_path.name

    def read_dataset_files(self, dataset_path: Path | None = None) -> tuple[list[str], list[dict[str, str]]]:
        """Reads all files from a dataset folder structure."""
        if dataset_path is None:
            dataset_path = ROOT_DIR.parent / "data" / "train"
        
        file_paths = []
        metadata_list = []

        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)

            if not os.path.isdir(subfolder_path):
                continue

            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                if not os.path.isfile(file_path):
                    continue

                file_paths.append(file_path)
                metadata_list.append({"document_type": subfolder})

        return file_paths, metadata_list

    async def populate_vector_db(self, options) -> bool:
        """Populates the vector database with document embeddings asynchronously."""
        try:
            logger.info("Starting vector database population...")

            self.split_dataset_single_test()
            file_paths, metadata = self.read_dataset_files()

            logger.info(f"Found {len(file_paths)} files to process")

            if not file_paths:
                logger.error("No files found to process")
                return False

            vector_db = VectorDBFactory.create("chromadb")
            vector_db.get_or_create_collection()

            ocr_engine = OCREngineFactory.create(options["ocr_engine"])

            docs = []
            successful_metadata = []
            batch_size = options["batch_size"]

            for i in range(0, len(file_paths), batch_size):
                batch_paths = file_paths[i : i + batch_size]
                batch_metadata = metadata[i : i + batch_size]
                batch_tasks = []

                self.stdout.write(f"Processing batch {i//batch_size + 1}/{(len(file_paths)-1)//batch_size + 1}")

                for file_path in batch_paths:
                    task = ocr_engine.extract_text_from_image_async(image_path=file_path)
                    batch_tasks.append(task)

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
                logger.error(
                    f"Mismatch between number of docs ({len(docs)}) and metadata ({len(successful_metadata)})"
                )
                return False

            vector_db.add_docs(docs, successful_metadata)
            logger.info(f"Successfully added {len(docs)} documents to vector database")
            return True

        except Exception as e:
            logger.error(f"Error populating vector database: {str(e)}")
            return False