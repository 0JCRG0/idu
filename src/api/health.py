from fastapi import APIRouter

from src.models import HealthCheck
from src.utils.logging_helper import get_custom_logger

router = APIRouter(tags=["Health Check"])
logger = get_custom_logger(__name__)


@router.get("/healthcheck", summary="Perform a Health Check", status_code=200, response_model=HealthCheck)
async def get_health() -> HealthCheck:
    """
    Health check endpoint.

    Returns
    -------
    str:
        A string to indicate that the service is up and running.
    """
    return HealthCheck(status="ok")
