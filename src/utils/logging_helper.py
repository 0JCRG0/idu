import logging

from tenacity import RetryCallState


def get_custom_logger(name: str) -> logging.Logger:
    """
    Get a custom logger with a specific name.

    Parameters
    ----------
    name: str
        The name of the logger.

    Returns
    -------
    logging.Logger
        The custom logger.
    """
    logging.basicConfig(level=logging.INFO)
    if not logging.getLogger(name).handlers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Add the formatter to the logger
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(fmt='%(levelname)s: %(asctime)s - "%(name)s" | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        )

        logger.addHandler(handler)
    else:
        logger = logging.getLogger(name)

    return logger


logger = get_custom_logger(__name__)


def log_attempt_retry(retry_state: RetryCallState):
    """
    Log the retry attempt.

    Parameters
    ----------
    retry_state: RetryCallState
        The state of the retry attempt.
    """
    if retry_state.attempt_number < 1:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING
    logger.log(
        loglevel,
        "Retrying %s: attempt %s ended with: %s",
        retry_state.fn,
        retry_state.attempt_number,
        retry_state.outcome,
    )
