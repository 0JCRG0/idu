from abc import ABC, abstractmethod


class OCREngineBase(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text_from_image(self, image_path: str, anchor: bool | None = None) -> str:
        """
        Extract text from an image using a specific OCR.

        Args:
            image_path (str): The path to the image file.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text from the image.
        """
        pass
