import ast
import json

import ell
from anthropic import Anthropic
from ell.lmp.complex import complex
from ell.types.message import system, user

from src.constants import ANTHROPIC_API_KEY, ELL_STORE_PATH, EXTRACTION_DEFAULT_MODEL
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)

ell.init(store=ELL_STORE_PATH, verbose=False)


@complex(
    model=EXTRACTION_DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=2000,
    client=Anthropic(api_key=ANTHROPIC_API_KEY),
)
def extract_entities_from_doc(system_prompt: str, user_content: str):
    """Extract entities from the document."""
    logger.info(f"Extracting entities from the document using '{EXTRACTION_DEFAULT_MODEL}'")
    return [
        system(system_prompt),
        user(user_content),
    ]


@complex(
    model=EXTRACTION_DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=2000,
    client=Anthropic(api_key=ANTHROPIC_API_KEY),
)
def validate_document_type(system_prompt: str, user_content: str):
    """Validate document type."""
    logger.info(f"Validating document type using '{EXTRACTION_DEFAULT_MODEL}'")
    return [
        system(system_prompt),
        user(user_content),
    ]


def extract_valid_json(response: str) -> dict:
    """Extract and return a valid JSON object from a given response string.

    Args:
        response (str): The JSON string to be validated and parsed.

    Returns
    -------
        dict[str, str]: The parsed JSON object.

    Raises
    ------
        json.JSONDecodeError: If no valid JSON object can be parsed.

    """
    # Try JSON first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try finding the brackets
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass

    # Try ast.literal_eval
    try:
        return ast.literal_eval(response)
    except (ValueError, SyntaxError):
        pass

    raise AssertionError("String is not a valid dictionary format")
