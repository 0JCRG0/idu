from pathlib import Path

import httpx
import pytest
import requests
from pydantic import ValidationError

from src.schemas.api import DocumentModelResponse
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


class TestExtractEntitiesIntegration:
    """Integration tests for entity extraction endpoints."""

    @pytest.fixture
    def api_url(self):
        """Base URL for the API - can be overridden by environment variable."""
        return "http://localhost:8000"

    @pytest.fixture
    def test_image_paths(self):
        """Retrieve all test image paths from the data/test folder."""
        test_folder = Path("data/test")
        file_paths = []

        for subfolder in test_folder.iterdir():
            if not subfolder.is_dir():
                continue

            for filename in subfolder.iterdir():
                file_paths.append(filename)

        if not file_paths:
            pytest.fail("No test images found in data/test")
        return file_paths

    def test_extract_entities_single_jpg(self, api_url, test_image_paths: list[Path]):
        """Test entity extraction from any image in test."""
        test_image_path = test_image_paths[0]
        doc_type = test_image_path.parent.name
        assert test_image_path.exists(), f"Test image not found: {test_image_path}"

        with open(test_image_path, "rb") as f:
            image_bytes = f.read()

        url = f"{api_url}/extract-entities/"
        files = {"file": (f"{doc_type}_{test_image_path.name}.jpg", image_bytes, "image/jpeg")}

        response = requests.post(url, files=files)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        try:
            DocumentModelResponse.model_validate(data)
            logger.info(f"Response validated successfully: {data}")
        except ValidationError as e:
            pytest.fail(f"Response validation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_extract_entities_multiple_files_async(self, api_url, test_image_paths: list[Path]):
        """Async test: entity extraction with two files sent in the same request."""
        assert len(test_image_paths) >= 2, "Need at least two test images for this test."
        first_image_path = test_image_paths[0]
        second_image_path = test_image_paths[1]

        assert first_image_path.exists(), f"First test image not found: {first_image_path}"
        assert second_image_path.exists(), f"Second test image not found: {second_image_path}"

        doc_type_1 = first_image_path.parent.name
        doc_type_2 = second_image_path.parent.name

        doc_1_bytes = first_image_path.read_bytes()
        doc_2_bytes = second_image_path.read_bytes()

        url = f"{api_url}/extract-entities/"

        async with httpx.AsyncClient(timeout=300.0) as client:
            files = [
                ("file", (f"{doc_type_1}_{first_image_path.name}", doc_1_bytes, "image/jpeg")),
                ("file", (f"{doc_type_2}_{second_image_path.name}", doc_2_bytes, "image/jpeg")),
            ]
            response = await client.post(url, files=files)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        assert isinstance(data, dict) and "files" in data, "Expected a dictionary with 'files' key"
        data = data["files"]

        for doc in data:
            try:
                DocumentModelResponse.model_validate(doc)
                logger.info(f"Response validated successfully: {doc}")
            except ValidationError as e:
                pytest.fail(f"Response validation failed for one of the files: {e}")

