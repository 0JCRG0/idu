import time

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)

from src.constants import DOCUMENT_FIELDS
from src.llm.llm import extract_entities_from_doc, extract_valid_json, validate_document_type
from src.llm.prompts import create_document_type_validation_prompt, create_extraction_prompt
from src.services.ocr.ocr import OCREngineFactory
from src.services.vector_db.vector_db import VectorDBFactory
from src.utils.logging_helper import get_custom_logger, log_attempt_retry

logger = get_custom_logger(__name__)


@retry(
    wait=wait_fixed(3) + wait_random(0, 2),
    reraise=True,
    stop=stop_after_attempt(3),
    retry=retry_if_not_exception_type(Exception),
    after=log_attempt_retry,
)
async def extract_entities_impl(image_input: bytes) -> dict:
    """
    Implement the endpoint for extraction of entities from the document.

    Args
    ----
        image_input (bytes): The image input as bytes.

    Returns
    -------
        dict: The response containing the extracted entities.
    """
    try:
        ocr_engine = OCREngineFactory.create()
        user_content = await ocr_engine.extract_text_from_image_async(image_input=image_input)
        logger.info(f"Extracted text: {user_content[:100]}...")
        start_time = time.time()

        vector_db = VectorDBFactory.create("chromadb")
        vector_db.get_or_create_collection()

        _, _, metadatas, _, confidence_scores = vector_db.find_similar_docs(user_content)

        document_type = metadatas[0]["document_type"]
        confidence = confidence_scores[0]

        logger.info(f"Document type: {document_type}, Confidence: {confidence}")

        document_type_validation_prompt = create_document_type_validation_prompt(document_type)
        validated_document_type = validate_document_type(
            document_type_validation_prompt, f"<document_text>{user_content}</document_text>"
        ).text  # type: ignore
        validated_document_type = validated_document_type.lower().strip()
        if validated_document_type not in DOCUMENT_FIELDS.keys():
            raise AssertionError("Document type validation failed")
        if validated_document_type != document_type:
            logger.warning(f"Document type validation mismatch: {document_type} != {validated_document_type}")
            logger.warning(f"Setting confidence to None and document_type to '{validated_document_type}'")
            confidence = None
            document_type = validated_document_type

        system_prompt = create_extraction_prompt(document_type)
        response = extract_entities_from_doc(system_prompt, f"<document_text>{user_content}</document_text>").text  # type: ignore
        response_json = extract_valid_json(response)
        result = {
            "document_type": document_type,
            "confidence": confidence,
            "entities": response_json,
            "processing_time": round(time.time() - start_time, 2),
        }
        return result
    except Exception as e:
        logger.error(f"Error extracting entities: {e}", exc_info=True)
        raise e
