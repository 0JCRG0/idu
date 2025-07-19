import io

from pdf2image import convert_from_bytes


def pdf_to_png_bytes(pdf_content: bytes) -> bytes:
    """
    Convert PDF content to PNG bytes.

    Parameters
    ----------
    pdf_content : bytes
        The PDF file content as bytes

    Returns
    -------
    bytes
        PNG image content as bytes (first page of PDF)

    Raises
    ------
    Exception
        If PDF conversion fails
    """
    try:
        images = convert_from_bytes(pdf_content, first_page=1, last_page=1, dpi=200)

        if not images:
            raise ValueError("PDF conversion resulted in no images")

        first_page = images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(f"Failed to convert PDF to PNG: {str(e)}") from e


def validate_and_convert_image(file_content: bytes, content_type: str, filename: str) -> bytes:
    """
    Validate and convert image file to a standardized format.

    Parameters
    ----------
    file_content : bytes
        The file content as bytes
    content_type : str
        The MIME type of the file
    filename : str
        The original filename

    Returns
    -------
    bytes
        Processed image content as bytes

    Raises
    ------
    ValueError
        If file format is not supported
    Exception
        If image processing fails
    """
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf") or content_type == "application/pdf":
        return pdf_to_png_bytes(file_content)

    if filename_lower.endswith(".png") or content_type == "image/png":
        return file_content

    if filename_lower.endswith((".jpg", ".jpeg")) or content_type in ["image/jpeg", "image/jpg"]:
        return file_content

    raise ValueError(f"Unsupported file format: {filename} (content-type: {content_type})")


def get_supported_extensions() -> tuple[str, ...]:
    """
    Get tuple of supported file extensions.

    Returns
    -------
    tuple[str, ...]
        Supported file extensions
    """
    return (".jpg", ".jpeg", ".png", ".pdf")


def get_supported_content_types() -> tuple[str, ...]:
    """
    Get tuple of supported MIME types.

    Returns
    -------
    tuple[str, ...]
        Supported MIME types
    """
    return ("image/jpeg", "image/jpg", "image/png", "application/pdf")
