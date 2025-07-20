from abc import ABC, abstractmethod


class OCREngineBase(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text_from_image(
        self, image_path: str | None = None, image_input: bytes | str | None = None, anchor: bool | None = None
    ) -> str:
        """
        Extract text from an image using a specific OCR.

        Args:
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | str | None, optional): The image input as bytes or a base64 string.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text from the image.
        """
        pass

    @abstractmethod
    async def extract_text_from_image_async(
        self, image_path: str | None = None, image_input: bytes | str | None = None, anchor: bool | None = None
    ) -> str:
        """
        Extract text from an image using the HF OCR model asynchronously.

        Args:
            image_path (str| None, optional): The path to the image file.
            image_input (bytes | str | None, optional): The image input as bytes or a base64 string.
            anchor (bool | None, optional): Whether to use an anchor for the OCR engine. Defaults to None.

        Returns
        -------
            str: The extracted text.
        """
        pass
