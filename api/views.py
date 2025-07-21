import asyncio

from django.http import JsonResponse
from django.shortcuts import render
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
    Extract entities from uploaded documents (JPG, PNG, or PDF).

    Supports both single file and multiple file uploads.

    Parameters
    ----------
    request : Request
        Django REST framework request object

    Returns
    -------
    Response
        The extracted entities and metadata for all files
    """
    try:
        files = request.FILES.getlist("file") or request.FILES.getlist("files") # type: ignore
        if not files:
            return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

        supported_extensions = get_supported_extensions()
        supported_content_types = get_supported_content_types()
        
        file_data = []
        
        for file in files:
            if not file.name:
                return Response({"error": "File must have a name"}, status=status.HTTP_400_BAD_REQUEST)

            if not file.name.lower().endswith(supported_extensions):
                return Response(
                    {
                        "error": f"Unsupported file extension for {file.name}. "
                        f"Supported formats: {', '.join(supported_extensions)}"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if file.content_type not in supported_content_types:
                supported_types_str = ", ".join(supported_content_types)
                return Response(
                    {
                        "error": f"Invalid content type for {file.name}: {file.content_type}. "
                        f"Supported types: {supported_types_str}"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            original_content = file.read()

            try:
                content = validate_and_convert_image(
                    original_content, file.content_type or "", file.name
                )
            except Exception as e:
                file_error = f"File processing error for {file.name}: {str(e)}"
                logger.error(file_error)
                return Response({"error": file_error}, status=status.HTTP_400_BAD_REQUEST)

            file_data.append({"content": content, "filename": file.name})
            logger.info(f"Queued for processing: {file.name}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tasks = [extract_entities_impl(file_info["content"]) for file_info in file_data]
            response_data_list = loop.run_until_complete(asyncio.gather(*tasks))
            
            results = []
            for i, response_data in enumerate(response_data_list):
                response_data["filename"] = file_data[i]["filename"]
                logger.info(response_data)
                validated_response = DocumentModelResponse.model_validate(response_data)
                results.append(validated_response.model_dump())
                
        finally:
            loop.close()

        if len(results) == 1:
            return Response(results[0], status=status.HTTP_200_OK)
        else:
            return Response({"files": results}, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def extract_entities_ui(request):
    """
    Render the UI for document entity extraction.

    Parameters
    ----------
    request : Request
        Django request object

    Returns
    -------
    HttpResponse
        Rendered HTML template
    """
    return render(request, "extract_entities.html")


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
