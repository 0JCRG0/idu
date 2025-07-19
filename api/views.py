import asyncio

from django.core.files.uploadedfile import UploadedFile
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

from src.core.orchestrator import extract_entities_impl
from src.schemas.api import DocumentModelResponse
from src.utils.file_processing import get_supported_content_types, get_supported_extensions, validate_and_convert_image
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)


def handle_error(exception: Exception) -> JsonResponse:
    """
    Handles errors by logging and returning JSON error response.

    Parameters
    ----------
    exception : Exception
        The exception to handle

    Returns
    -------
    JsonResponse
        JSON response with error details
    """
    logger.error(f"Error: {exception}", exc_info=True)
    return JsonResponse({"error": str(exception)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@parser_classes([MultiPartParser])
def extract_entities(request: Request) -> Response:
    """
    Extract entities from uploaded document (JPG, PNG, or PDF).

    Parameters
    ----------
    request : Request
        Django REST framework request object

    Returns
    -------
    Response
        The extracted entities and metadata
    """
    try:
        file: UploadedFile | None = request.FILES.get("file")  # type: ignore
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate file type
        if not file.name:
            return Response({"error": "File must have a name"}, status=status.HTTP_400_BAD_REQUEST)

        supported_extensions = get_supported_extensions()
        supported_content_types = get_supported_content_types()

        if not file.name.lower().endswith(supported_extensions):
            return Response(
                {"error": f"Unsupported file extension. Supported formats: {', '.join(supported_extensions)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if file.content_type not in supported_content_types:
            supported_types_str = ", ".join(supported_content_types)
            return Response(
                {"error": f"Invalid content type: {file.content_type}. Supported types: {supported_types_str}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Read and process file content
        original_content = file.read()

        # Convert file to appropriate format (handles PDF to PNG conversion)
        try:
            content = validate_and_convert_image(original_content, file.content_type or "", file.name)
        except Exception as e:
            return Response({"error": f"File processing error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(f"Processing file: {file.name}")

        # Since Django views are synchronous by default, we need to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_data = loop.run_until_complete(extract_entities_impl(content))
        finally:
            loop.close()

        logger.info(response_data)

        # Validate response with Pydantic model
        validated_response = DocumentModelResponse.model_validate(response_data)

        return Response(validated_response.model_dump(), status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["GET"])
def health_check(request: Request) -> Response:
    """
    Health check endpoint.

    Parameters
    ----------
    request : Request
        Django REST framework request object

    Returns
    -------
    Response
        Health check response
    """
    return Response({"status": "ok"}, status=status.HTTP_200_OK)
