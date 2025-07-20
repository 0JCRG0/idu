from pathlib import Path

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
