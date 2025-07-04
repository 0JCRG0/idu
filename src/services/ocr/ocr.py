from typing import Literal

from src.services.ocr.base import OCREngineBase
from src.services.ocr.olmo_ocr_impl import OlmoOCREngine
from src.services.ocr.tesseract_impl import TesseractOCREngine


class OCREngineFactory:
    """Factory class for creating OCR engine instances."""
    
    @staticmethod
    def create(engine_type: Literal["tesseract", "olmo_ocr"] = "olmo_ocr") -> OCREngineBase:
        """
        Create an OCR engine instance based on the specified type.
        
        Args:
            engine_type: Type of OCR engine ("tesseract" or "olmo_ocr")
            
        Returns
        -------
            Instance of the requested OCR engine implementation
            
        Raises
        ------
            ValueError: If an unsupported engine type is specified
        """
        if engine_type == "tesseract":
            return TesseractOCREngine()
        elif engine_type == "olmo_ocr":
            return OlmoOCREngine()
        else:
            raise ValueError(f"Unsupported OCR engine type: {engine_type}")