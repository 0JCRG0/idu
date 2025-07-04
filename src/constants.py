from src.utils.env_helper import EnvHelper
from pathlib import Path

env = EnvHelper.load_env_variables()
ROOT_DIR = Path(__file__).parent
OPENAI_API_KEY = env.api_keys.openai
ANTHROPIC_API_KEY = env.api_keys.anthropic
HF_SECRETS = env.api_keys.hf
EMBEDDING_DEFAULT_MODEL = "text-embedding-3-small"
EXTRACTION_DEFAULT_MODEL = "claude-4-sonnet-20250514"

DOCUMENT_FIELDS = {
    "letter": [
        {"name": "sender_name", "description": "Name of the person or organization sending the letter"},
        {"name": "recipient_name", "description": "Name of the person or organization receiving the letter"},
        {"name": "date", "description": "Date the letter was written"},
        {"name": "subject", "description": "Subject or RE line of the letter"},
        {"name": "salutation", "description": "Opening greeting (e.g., 'Dear Mr. Smith')"}
    ],
    "specification": [
        {"name": "document_title", "description": "Title of the specification document"},
        {"name": "version_number", "description": "Version or revision number"},
        {"name": "effective_date", "description": "Date when the specification becomes effective"},
        {"name": "requirements", "description": "List of key requirements or specifications"},
        {"name": "compliance_standards", "description": "Referenced standards or regulations"}
    ],
    "handwritten": [
        {"name": "author_name", "description": "Name of the person who wrote the document (if identifiable)"},
        {"name": "date_written", "description": "Date the document was written"},
        {"name": "document_type", "description": "Type of handwritten document (note, letter, form, etc.)"},
        {"name": "main_content", "description": "Primary text content of the document"},
        {"name": "legibility_notes", "description": "Any illegible sections or unclear text"}
    ],
    "presentation": [
        {"name": "presentation_title", "description": "Title of the presentation"},
        {"name": "presenter_name", "description": "Name(s) of the presenter(s)"},
        {"name": "presentation_date", "description": "Date of the presentation"},
        {"name": "slide_count", "description": "Total number of slides"},
        {"name": "key_topics", "description": "List of main topics or sections covered"}
    ],
    "resume": [
        {"name": "candidate_name", "description": "Full name of the candidate"},
        {"name": "contact_information", "description": "Email, phone, address, and other contact details"},
        {"name": "work_experience", "description": "List of previous positions with company names, titles, and dates"},
        {"name": "education", "description": "Educational background including degrees, institutions, and dates"},
        {"name": "skills", "description": "List of technical and soft skills"}
    ],
    "budget": [
        {"name": "budget_period", "description": "Time period covered by the budget (e.g., 'FY 2024', 'Q1 2024')"},
        {"name": "total_budget", "description": "Total budget amount"},
        {"name": "department_name", "description": "Department or organization name"},
        {"name": "line_items", "description": "List of budget categories with allocated amounts"},
        {"name": "approval_status", "description": "Whether the budget is draft, proposed, or approved"}
    ],
    "email": [
        {"name": "sender_email", "description": "Email address of the sender"},
        {"name": "recipient_emails", "description": "List of recipient email addresses (To, CC, BCC)"},
        {"name": "subject_line", "description": "Subject line of the email"},
        {"name": "date_sent", "description": "Date and time the email was sent"},
        {"name": "attachments", "description": "List of attachment names if any"}
    ],
    "scientific_publication": [
        {"name": "title", "description": "Title of the scientific paper"},
        {"name": "authors", "description": "List of all authors with their affiliations"},
        {"name": "abstract", "description": "Abstract or summary of the publication"},
        {"name": "keywords", "description": "List of keywords or key terms"},
        {"name": "doi", "description": "Digital Object Identifier if available"}
    ],
    "invoice": [
        {"name": "invoice_number", "description": "Unique invoice identifier"},
        {"name": "invoice_date", "description": "Date the invoice was issued"},
        {"name": "due_date", "description": "Payment due date"},
        {"name": "vendor_details", "description": "Vendor/supplier name, address, and contact information"},
        {"name": "total_amount", "description": "Total amount due including taxes"}
    ],
    "file_folder": [
        {"name": "folder_title", "description": "Title or label on the folder"},
        {"name": "date_range", "description": "Date range of documents contained (if specified)"},
        {"name": "file_count", "description": "Number of files or documents contained"},
        {"name": "category", "description": "Category or classification of the folder"},
        {"name": "reference_number", "description": "Any reference or catalog number"}
    ],
    "memo": [
        {"name": "to", "description": "Recipient(s) of the memo"},
        {"name": "from", "description": "Sender of the memo"},
        {"name": "date", "description": "Date of the memo"},
        {"name": "subject", "description": "Subject line of the memo"},
        {"name": "action_items", "description": "List of action items or next steps mentioned"}
    ],
    "scientific_report": [
        {"name": "report_title", "description": "Title of the scientific report"},
        {"name": "principal_investigator", "description": "Name of the principal investigator or lead researcher"},
        {"name": "institution", "description": "Research institution or organization"},
        {"name": "report_date", "description": "Date the report was issued"},
        {"name": "key_findings", "description": "List of major findings or conclusions"}
    ],
    "form": [
        {"name": "form_title", "description": "Title or name of the form"},
        {"name": "form_number", "description": "Form number or identifier"},
        {"name": "completion_date", "description": "Date the form was completed"},
        {"name": "filled_fields", "description": "Dictionary of field names and their filled values"},
        {"name": "signature_present", "description": "Whether the form has been signed"}
    ],
    "advertisement": [
        {"name": "product_name", "description": "Name of the product or service being advertised"},
        {"name": "company_name", "description": "Name of the advertising company"},
        {"name": "headline", "description": "Main headline or slogan"},
        {"name": "call_to_action", "description": "Primary call to action (e.g., 'Buy Now', 'Learn More')"},
        {"name": "contact_information", "description": "Phone, website, or other contact details"}
    ],
    "questionnaire": [
        {"name": "questionnaire_title", "description": "Title of the questionnaire"},
        {"name": "respondent_info", "description": "Information about the respondent (if not anonymous)"},
        {"name": "completion_date", "description": "Date the questionnaire was completed"},
        {"name": "questions_and_answers", "description": "List of questions with their corresponding answers"},
        {"name": "total_questions", "description": "Total number of questions in the questionnaire"}
    ],
    "news_article": [
        {"name": "headline", "description": "Article headline"},
        {"name": "author", "description": "Author or journalist name"},
        {"name": "publication_date", "description": "Date of publication"},
        {"name": "news_source", "description": "Name of the newspaper or news organization"},
        {"name": "article_summary", "description": "Brief summary of the article's main points"}
    ]
}