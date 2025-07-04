import os
import random
import shutil
from pathlib import Path

from src.constants import ROOT_DIR
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


def split_dataset(
    source_folder: Path = ROOT_DIR.parent / "data" / "docs-sm",
    output_folder: Path = ROOT_DIR.parent / "data",
    train_ratio: float = 0.02,
    random_seed: None | int = 42,
    train_folder_name: str = "train",
    test_folder_name: str = "test",
) -> tuple[str, str]:
    """
    Split a dataset folder into train and test sets.

    Args
    ----
        source_folder: Path to the dataset folder containing subfolders with files (default: "data/docs-sm")
        output_folder: Base output directory (default: "data")
        train_ratio: Ratio of files to include in training set (default: 0.6)
        random_seed: Random seed for reproducibility (default: 42)
        train_folder_name: Name for training set folder (default: "train")
        test_folder_name: Name for test set folder (default: "test")

    Returns
    -------
        tuple of (train_path, test_path)
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Create output directories
    output_path = output_folder
    train_path = output_path / train_folder_name
    test_path = output_path / test_folder_name

    # Clean and create directories
    for path in [train_path, test_path]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    source_path = source_folder

    # Process each subfolder in the dataset
    for subfolder in source_path.iterdir():
        if not subfolder.is_dir():
            continue

        # Create corresponding subfolders in train and test
        train_subfolder = train_path / subfolder.name
        test_subfolder = test_path / subfolder.name
        train_subfolder.mkdir(exist_ok=True)
        test_subfolder.mkdir(exist_ok=True)

        # Get all files in the subfolder
        files = [f for f in subfolder.iterdir() if f.is_file()]

        # Shuffle files for random split
        random.shuffle(files)

        # Calculate split point
        split_point = int(len(files) * train_ratio)
        train_files = files[:split_point]
        test_files = files[split_point:]

        # Copy files to respective folders
        for file in train_files:
            shutil.copy2(file, train_subfolder / file.name)

        for file in test_files:
            shutil.copy2(file, test_subfolder / file.name)

        logger.info(f"Processed {subfolder.name}: {len(train_files)} train, {len(test_files)} test")

    logger.info("\nDataset split complete!")
    logger.info(f"Train set: {train_path.name}")
    logger.info(f"Test set: {test_path.name}")

    return train_path.name, test_path.name


def read_dataset_files(
    dataset_path: Path = ROOT_DIR.parent / "data" / "train",
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Reads all files from a dataset folder structure.

    Args:
        dataset_path: Path to the dataset folder containing subfolders

    Returns
    -------
        tuple of (file_paths_list, metadata_list)
        where metadata_list contains {'document_type': subfolder_name} dicts
    """
    file_paths = []
    metadata_list = []

    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)

        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Iterate through all files in the subfolder
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)

            # Skip if not a file
            if not os.path.isfile(file_path):
                continue

            file_paths.append(file_path)
            metadata_list.append({"document_type": subfolder})

    return file_paths, metadata_list
