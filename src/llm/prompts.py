import json

from src.constants import DOCUMENT_FIELDS


# Prompt from https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py
def prompt_olmocr_with_anchor(base_text: str) -> str:
    """Returns the prompt for the finetuning step of the OLMOCR model."""
    return (
        f"Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. "  # noqa: E501
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


def default_olmocr_prompt() -> str:
    """Returns the default prompt for the OLMOCR model."""
    return "Just return the plain text representation of this document as if you were reading it naturally."


def format_field_list(fields: list[dict[str, str]]) -> str:
    """Format the field list for inclusion in the prompt."""
    formatted_fields = []
    for field in fields:
        formatted_fields.append(f"    - {field['name']}: {field['description']}")
    return "\n".join(formatted_fields)


def create_document_type_validation_prompt(current_document_type: str) -> str:
    """
    Create a prompt to validate if the detected document type is appropriate.

    Args:
        document_text: The actual text content of the document
        current_document_type: The currently selected document type

    Returns
    -------
        A formatted prompt string for document type validation
    """
    # Format confidence scores with their fields
    type_descriptions = []
    for doc_type in DOCUMENT_FIELDS:
        fields_summary = ", ".join([f["name"] for f in DOCUMENT_FIELDS[doc_type]])
        if len(DOCUMENT_FIELDS[doc_type]) > 5:
            fields_summary += f", ... ({len(DOCUMENT_FIELDS[doc_type])} fields total)"
        type_descriptions.append(f"- {doc_type} \n  Fields: {fields_summary} \n\n\n")

    available_document_types = "\n".join(type_descriptions)

    return f"""
    <document_type_validation_task>
        <context>
            <current_selection>{current_document_type}</current_selection>
            <objective>
            Validate if the current document type selection is appropriate based on the <available_document_types> and the <document_text> provided by the user.
            </objective>
        </context>

        <instructions>
            <requirement>Review the document content and the document fields for each document type.</requirement>
            <requirement>If the current document type selection seems appropriate, return the <current_selection> value. </requirement>
            <requirement>If a different document type would be more appropriate, respond with ONLY that document type name.</requirement>
            <requirement>Consider whether the document content matches the expected fields.</requirement>
            <requirement>Your response must be a single word - either the <current_selection> value or the alternative document type name.</requirement>
        </instructions>

        <available_document_types>
{available_document_types}
        </available_document_types>

        <response_format>
            - Single word only
            - Either the <current_selection> value or an alternative document type name
            - No explanations, no additional text
        </response_format>
    </document_type_validation_task>"""  # noqa: E501


def create_extraction_prompt(document_type: str, custom_fields: list[dict[str, str]] | None = None) -> str:
    """
    Create a structured document extraction prompt with XML tags based on the document type.

    Args:
        document_type: The type of document being processed
        custom_fields: Optional custom fields to extract instead of the default ones

    Returns
    -------
        A formatted prompt string ready for use with an LLM

    Raises
    ------
        ValueError: If document_type is not recognized and no custom_fields provided
    """
    # Use custom fields if provided, otherwise load from predefined fields
    if custom_fields:
        fields = custom_fields
    else:
        if document_type not in DOCUMENT_FIELDS:
            raise ValueError(
                f"Unknown document type: {document_type}. Known types: {', '.join(DOCUMENT_FIELDS.keys())}"
            )
        fields = DOCUMENT_FIELDS[document_type]

    field_list = format_field_list(fields)

    example_json = {}
    for field in fields:
        if "list" in field["description"].lower():
            example_json[field["name"]] = ["example_item1", "example_item2"]
        elif "date" in field["description"].lower():
            example_json[field["name"]] = "2024-01-15"
        elif "number" in field["name"] or "count" in field["name"] or "amount" in field["name"]:
            example_json[field["name"]] = 123
        else:
            example_json[field["name"]] = "extracted_value"

    # Format example JSON with proper indentation
    example_json_str = json.dumps(example_json, indent=6).replace("\n", "\n    ")

    return f"""
    <document_extraction_task>
        <context>
            <document_type>{document_type}</document_type>
            <extraction_objective>
            Extract specific structured data from the <document_text> that will be provided by the user and return it in a standardized JSON format.
            </extraction_objective>
        </context>

        <instructions>
            <requirement>You MUST extract ALL of the following fields from the document text below.</requirement>
            <requirement>Each field MUST be included in your response, even if the value is null or empty.</requirement>
            <requirement>You MUST return ONLY a valid JSON object with no additional text, explanation, or markdown formatting.</requirement>
            <requirement>Extract values exactly as they appear in the document without interpretation unless explicitly required by field type.</requirement>
        </instructions>

        <output_format>
            <format_type>JSON</format_type>
            <format_requirements>
            - Valid JSON syntax only
            - No markdown code blocks
            - No explanatory text before or after
            - No comments within JSON
            - Use null for missing values
            - Maintain original data types (strings as strings, numbers as numbers)
            </format_requirements>
        </output_format>

        <fields_to_extract>
            {field_list}
        </fields_to_extract>

        <example_response_format>
            {example_json_str}
        </example_response_format>
        

    </document_extraction_task>"""  # noqa: E501
