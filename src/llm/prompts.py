# Prompt from https://github.com/allenai/olmocr/blob/main/olmocr/prompts/prompts.py
def prompt_olmocr_with_anchor(base_text: str) -> str:
    """Returns the prompt for the finetuning step of the OLMOCR model."""
    return (
        f"Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


def default_olmocr_prompt() -> str:
    """Returns the default prompt for the OLMOCR model."""
    return "Just return the plain text representation of this document as if you were reading it naturally."
