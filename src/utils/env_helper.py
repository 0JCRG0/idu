import os

from dotenv import find_dotenv, load_dotenv

from src.models import APIKeys, EnvVariables
from src.utils.logging_helper import get_custom_logger

logger = get_custom_logger(__name__)

class EnvHelper:
    """EnvHelper loads environment variables from a .env file."""

    @classmethod
    def load_env_variables(cls, env_filename: str = ".env") -> EnvVariables:
        """Load environment variables from a .env file."""
        env_file = find_dotenv(filename=env_filename, usecwd=True)
        if env_file:
            logger.info("Loading environment variables from %s", env_file)
            load_dotenv(dotenv_path=env_file, verbose=True, override=True)
        else:
            logger.error(f"Variables could not be loaded from the '{env_filename}'")
            raise AssertionError(f"Environment should be loaded from {env_filename}")

        return EnvVariables(
            api_keys=APIKeys(
                openai=os.environ["OPENAI_API_KEY"],
                anthropic=os.environ["ANTHROPIC_API_KEY"],
            )
        )
