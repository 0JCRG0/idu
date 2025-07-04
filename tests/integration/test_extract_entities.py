from pathlib import Path

import pytest
import requests


class TestExtractEntitiesIntegration:
    """Integration tests for entity extraction endpoints."""

    @pytest.fixture
    def api_url(self):
        """Base URL for the API - can be overridden by environment variable."""
        return "http://localhost:8000"

    @pytest.fixture
    def test_image_path(self):
        """Path to test image."""
        return Path("data/test/advertisement/660202.jpg")

    def test_extract_entities_from_advertisement(self, api_url, test_image_path):
        """Test entity extraction from advertisement image."""
        # Ensure test file exists
        assert test_image_path.exists(), f"Test image not found: {test_image_path}"

        # Read image file
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()

        # Prepare request
        url = f"{api_url}/v1/extract-entities"
        files = {"file": ("advertisement.jpg", image_bytes, "image/jpeg")}

        # Make request
        response = requests.post(url, files=files)

        # Assert response status
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        # Parse response
        data = response.json()

        assert data["document_type"] == "advertisement"
