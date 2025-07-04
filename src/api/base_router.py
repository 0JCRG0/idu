from collections.abc import Callable
from enum import Enum
from typing import Any

from fastapi import APIRouter

from src.models import SuccessfulResponse


class BaseRouter:
    """Base router class."""

    def __init__(
        self,
        path: str,
        endpoint: Callable,
        tags: list[str | Enum],
        prefix: str = "/v1",
        response_model: Any | None = None,
    ):
        """
        Initialize a base router with specific functionality.

        Parameters
        ----------
        path : str
            The path for the router.
        endpoint : Callable
            The endpoint function to handle requests.
        tags : list[str | Enum]
            The tags for the router.
        prefix : str, optional
            The prefix for the router, by default "v1"
        response_model : Optional[Any], optional
            Custom response model for the endpoint, by default None
        """
        self.prefix = prefix
        self.endpoint = endpoint
        self.path = path
        self.tags = tags
        self.response_model = response_model or SuccessfulResponse

        self.router = APIRouter(prefix=self.prefix, tags=self.tags)
        self.__setup_routes()

    def __setup_routes(self) -> None:
        """Set up API routes for the router."""
        self.router.add_api_route(
            self.path,
            self.endpoint,
            methods=["POST"],
            status_code=200,
            response_model=self.response_model,
        )
