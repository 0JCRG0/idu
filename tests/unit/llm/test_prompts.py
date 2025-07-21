import pytest

from src.constants import DOCUMENT_FIELDS
from src.llm.prompts import (
    create_document_type_validation_prompt,
    create_extraction_prompt,
    default_olmocr_prompt,
    format_field_list,
    prompt_olmocr_with_anchor,
)


class TestPromptOlmOcrWithAnchor:
    """Tests for the prompt_olmocr_with_anchor function."""

    def test_prompt_olmocr_with_anchor_basic(self):
        """Test basic prompt generation with anchor for OLM OCR."""
        base_text = "This is sample text"
        result = prompt_olmocr_with_anchor(base_text)

        assert "Below is the image of one page of a document" in result
        assert f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END" in result
        assert "Do not hallucinate" in result

    def test_prompt_olmocr_with_anchor_empty_text(self):
        """Test prompt generation with empty base text."""
        result = prompt_olmocr_with_anchor("")
        assert "RAW_TEXT_START\n\nRAW_TEXT_END" in result


class TestDefaultOlmOcrPrompt:
    """Tests for the default_olmocr_prompt function."""

    def test_default_olmocr_prompt(self):
        """Test the default OLM OCR prompt output."""
        result = default_olmocr_prompt()
        expected = "Just return the plain text representation of this document as if you were reading it naturally."
        assert result == expected


class TestFormatFieldList:
    """Test cases for the format_field_list function."""

    def test_format_field_list_single_field(self):
        """Test format_field_list with a single field."""
        fields = [{"name": "field1", "description": "Description 1"}]
        result = format_field_list(fields)
        assert result == "    - field1: Description 1"

    def test_format_field_list_multiple_fields(self):
        """Test format_field_list with a multiple fields."""
        fields = [
            {"name": "field1", "description": "Description 1"},
            {"name": "field2", "description": "Description 2"},
        ]
        result = format_field_list(fields)
        expected = "    - field1: Description 1\n    - field2: Description 2"
        assert result == expected


class TestCreateDocumentTypeValidationPrompt:
    """Test cases for the create_document_type_validation_prompt function."""

    def test_create_document_type_validation_prompt_basic(self):
        """Test document type validation prompt with basic input."""
        result = create_document_type_validation_prompt("invoice")

        expected_result = "\n    <document_type_validation_task>\n        <context>\n            <current_selection>invoice</current_selection>\n            <objective>\n            Validate if the current document type selection is appropriate based on the <available_document_types> and the <document_text> provided by the user.\n            </objective>\n        </context>\n\n        <instructions>\n            <requirement>Review the document content and the document fields for each document type.</requirement>\n            <requirement>If the current document type selection seems appropriate, return the <current_selection> value. </requirement>\n            <requirement>If a different document type would be more appropriate, respond with ONLY that document type name.</requirement>\n            <requirement>Consider whether the document content matches the expected fields.</requirement>\n            <requirement>Your response must be a single word - either the <current_selection> value or the alternative document type name.</requirement>\n        </instructions>\n\n        <available_document_types>\n- letter \n  Fields: sender_name, recipient_name, date, subject, salutation \n\n\n\n- specification \n  Fields: document_title, version_number, effective_date, requirements, compliance_standards \n\n\n\n- handwritten \n  Fields: author_name, date_written, document_type, main_content, legibility_notes \n\n\n\n- presentation \n  Fields: presentation_title, presenter_name, presentation_date, slide_count, key_topics \n\n\n\n- resume \n  Fields: candidate_name, contact_information, work_experience, education, skills \n\n\n\n- budget \n  Fields: budget_period, total_budget, department_name, line_items, approval_status \n\n\n\n- email \n  Fields: sender_email, recipient_emails, subject_line, date_sent, attachments \n\n\n\n- scientific_publication \n  Fields: title, authors, abstract, keywords, doi \n\n\n\n- invoice \n  Fields: invoice_number, invoice_date, due_date, vendor_details, total_amount \n\n\n\n- file_folder \n  Fields: folder_title, date_range, file_count, category, reference_number \n\n\n\n- memo \n  Fields: to, from, date, subject, action_items \n\n\n\n- scientific_report \n  Fields: report_title, principal_investigator, institution, report_date, key_findings \n\n\n\n- form \n  Fields: form_title, form_number, completion_date, filled_fields, signature_present \n\n\n\n- advertisement \n  Fields: product_name, company_name, headline, call_to_action, contact_information \n\n\n\n- questionnaire \n  Fields: questionnaire_title, respondent_info, completion_date, questions_and_answers, total_questions \n\n\n\n- news_article \n  Fields: headline, author, publication_date, news_source, article_summary \n\n\n\n        </available_document_types>\n\n        <response_format>\n            - Single word only\n            - Either the <current_selection> value or an alternative document type name\n            - No explanations, no additional text\n        </response_format>\n    </document_type_validation_task>"  # noqa: E501

        assert result == expected_result

    def test_create_document_type_validation_prompt_different_types(self):
        """Test validation prompt for different document types."""
        for doc_type in ["letter", "memo", "resume"]:
            result = create_document_type_validation_prompt(doc_type)
            assert f"<current_selection>{doc_type}</current_selection>" in result


class TestCreateExtractionPrompt:
    """Tests for the create_extraction_prompt function."""

    def test_create_extraction_prompt_valid_document_type(self):
        """Test extraction prompt for a valid document type."""
        result = create_extraction_prompt("invoice")

        expected_result = '\n    <document_extraction_task>\n        <context>\n            <document_type>invoice</document_type>\n            <extraction_objective>\n            Extract specific structured data from the <document_text> that will be provided by the user and return it in a standardized JSON format.\n            </extraction_objective>\n        </context>\n\n        <instructions>\n            <requirement>You MUST extract ALL of the following fields from the document text below.</requirement>\n            <requirement>Each field MUST be included in your response, even if the value is null or empty.</requirement>\n            <requirement>You MUST return ONLY a valid JSON object with no additional text, explanation, or markdown formatting.</requirement>\n            <requirement>Extract values exactly as they appear in the document without interpretation unless explicitly required by field type.</requirement>\n        </instructions>\n\n        <output_format>\n            <format_type>JSON</format_type>\n            <format_requirements>\n            - Valid JSON syntax only\n            - No markdown code blocks\n            - No explanatory text before or after\n            - No comments within JSON\n            - Use null for missing values\n            - Maintain original data types (strings as strings, numbers as numbers)\n            </format_requirements>\n        </output_format>\n\n        <fields_to_extract>\n                - invoice_number: Unique invoice identifier\n    - invoice_date: Date the invoice was issued\n    - due_date: Payment due date\n    - vendor_details: Vendor/supplier name, address, and contact information\n    - total_amount: Total amount due including taxes\n        </fields_to_extract>\n\n        <example_response_format>\n            {\n          "invoice_number": 123,\n          "invoice_date": "2024-01-15",\n          "due_date": "2024-01-15",\n          "vendor_details": "extracted_value",\n          "total_amount": 123\n    }\n        </example_response_format>\n        \n\n    </document_extraction_task>'  # noqa: E501

        assert result == expected_result

    def test_create_extraction_prompt_invalid_document_type(self):
        """Test extraction prompt raises error for invalid type."""
        with pytest.raises(ValueError, match="Unknown document type: invalid_type"):
            create_extraction_prompt("invalid_type")

    def test_create_extraction_prompt_custom_fields(self):
        """Test extraction prompt with custom fields."""
        custom_fields = [
            {"name": "custom_field1", "description": "Custom description 1"},
            {"name": "custom_field2", "description": "Custom description 2"},
        ]
        result = create_extraction_prompt("any_type", custom_fields)

        assert "custom_field1: Custom description 1" in result
        assert "custom_field2: Custom description 2" in result
        assert "<document_type>any_type</document_type>" in result

    def test_create_extraction_prompt_example_json_generation(self):
        """Test if example JSON is generated for all custom field types."""
        custom_fields = [
            {"name": "date_field", "description": "A date field"},
            {"name": "list_field", "description": "A list of items"},
            {"name": "number_field", "description": "A number"},
            {"name": "regular_field", "description": "A regular field"},
            {"name": "amount_field", "description": "An amount"},
            {"name": "count_items", "description": "Count of items"},
        ]
        result = create_extraction_prompt("test", custom_fields)

        # Check for the JSON structure patterns, accounting for indentation
        assert "date_field" in result and "2024-01-15" in result
        assert "list_field" in result and "example_item1" in result
        assert "number_field" in result and "123" in result
        assert "regular_field" in result and "extracted_value" in result
        assert "amount_field" in result and "123" in result
        assert "count_items" in result and "123" in result

    def test_create_extraction_prompt_json_formatting(self):
        """Test that the example JSON in prompt is formatted properly."""
        custom_fields = [{"name": "test_field", "description": "Test field"}]
        result = create_extraction_prompt("test", custom_fields)

        example_section_start = result.find("<example_response_format>")
        example_section_end = result.find("</example_response_format>")
        example_content = result[example_section_start:example_section_end]

        assert '"test_field": "extracted_value"' in example_content

    def test_create_extraction_prompt_all_document_types(self):
        """Test extraction prompt can be created for all document types."""
        for doc_type in DOCUMENT_FIELDS.keys():
            result = create_extraction_prompt(doc_type)
            assert f"<document_type>{doc_type}</document_type>" in result
            assert "JSON" in result
            assert len(result) > 500

    def test_create_extraction_prompt_field_detection_logic(self):
        """Test default value logic for special field types in example JSON."""
        test_cases = [
            ({"name": "birth_date", "description": "Date of birth"}, ("birth_date", "2024-01-15")),
            ({"name": "skills_list", "description": "List of skills"}, ("skills_list", "example_item1")),
            ({"name": "total_amount", "description": "Total amount"}, ("total_amount", "123")),
            ({"name": "item_count", "description": "Number of items"}, ("item_count", "123")),
            ({"name": "salary_number", "description": "Salary number"}, ("salary_number", "123")),
            ({"name": "regular_field", "description": "Just a field"}, ("regular_field", "extracted_value")),
        ]

        for field, (expected_field, expected_value) in test_cases:
            result = create_extraction_prompt("test", [field])
            assert expected_field in result and expected_value in result
