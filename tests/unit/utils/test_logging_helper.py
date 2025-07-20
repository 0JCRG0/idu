import logging
from unittest.mock import Mock, patch

from tenacity import RetryCallState

from src.utils.logging_helper import get_custom_logger, log_attempt_retry


class TestGetCustomLogger:
    """Unit tests for the get_custom_logger function."""

    def test_get_custom_logger_new_logger(self):
        """Test creating a new custom logger."""
        with (
            patch("logging.basicConfig") as mock_basicConfig,
            patch("logging.getLogger") as mock_getLogger,
            patch("logging.StreamHandler") as mock_StreamHandler,
            patch("logging.Formatter") as mock_Formatter,
        ):
            mock_logger = Mock()
            mock_logger.handlers = []
            mock_getLogger.return_value = mock_logger

            mock_handler = Mock()
            mock_StreamHandler.return_value = mock_handler

            mock_formatter = Mock()
            mock_Formatter.return_value = mock_formatter

            result = get_custom_logger("test_logger")

            assert result == mock_logger
            mock_basicConfig.assert_called_once_with(level=logging.INFO)
            mock_getLogger.assert_called_with("test_logger")
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
            assert mock_logger.propagate is False
            mock_handler.setFormatter.assert_called_once_with(mock_formatter)
            mock_logger.addHandler.assert_called_once_with(mock_handler)

    def test_get_custom_logger_formatter_configuration(self):
        """Test that the formatter is configured correctly."""
        with (
            patch("logging.basicConfig"),
            patch("logging.getLogger") as mock_getLogger,
            patch("logging.StreamHandler") as mock_StreamHandler,
            patch("logging.Formatter") as mock_Formatter,
        ):
            mock_logger = Mock()
            mock_logger.handlers = []
            mock_getLogger.return_value = mock_logger

            mock_handler = Mock()
            mock_StreamHandler.return_value = mock_handler

            mock_formatter = Mock()
            mock_Formatter.return_value = mock_formatter

            get_custom_logger("test_logger")

            mock_Formatter.assert_called_once_with(
                fmt='%(levelname)s: %(asctime)s - "%(name)s" | %(message)s', datefmt="%Y-%m-%d %H:%M:%S"
            )

    def test_get_custom_logger_multiple_calls_same_name(self):
        """Test multiple calls with the same logger name."""
        with patch("logging.basicConfig"), patch("logging.getLogger") as mock_getLogger:
            mock_logger = Mock()
            # First call - no handlers, second call - has handlers
            mock_logger.handlers = []
            mock_getLogger.return_value = mock_logger

            # First call should set up the logger
            result1 = get_custom_logger("same_name")

            # Simulate handlers being added
            mock_logger.handlers = [Mock()]

            # Second call should return existing logger
            result2 = get_custom_logger("same_name")

            assert result1 == mock_logger
            assert result2 == mock_logger


class TestLogAttemptRetry:
    """Unit tests for the log_attempt_retry function."""

    @patch("src.utils.logging_helper.logger")
    def test_log_attempt_retry_first_attempt(self, mock_logger):
        """Test logging for first retry attempt (attempt_number < 1)."""
        mock_retry_state = Mock(spec=RetryCallState)
        mock_retry_state.attempt_number = 0
        mock_retry_state.fn = "test_function"
        mock_retry_state.outcome = "success"

        log_attempt_retry(mock_retry_state)

        mock_logger.log.assert_called_once_with(
            logging.INFO, "Retrying %s: attempt %s ended with: %s", "test_function", 0, "success"
        )

    @patch("src.utils.logging_helper.logger")
    def test_log_attempt_retry_subsequent_attempt(self, mock_logger):
        """Test logging for subsequent retry attempts (attempt_number >= 1)."""
        mock_retry_state = Mock(spec=RetryCallState)
        mock_retry_state.attempt_number = 2
        mock_retry_state.fn = "failing_function"
        mock_retry_state.outcome = "failed"

        log_attempt_retry(mock_retry_state)

        mock_logger.log.assert_called_once_with(
            logging.WARNING, "Retrying %s: attempt %s ended with: %s", "failing_function", 2, "failed"
        )

    @patch("src.utils.logging_helper.logger")
    def test_log_attempt_retry_boundary_case(self, mock_logger):
        """Test logging for boundary case (attempt_number = 1)."""
        mock_retry_state = Mock(spec=RetryCallState)
        mock_retry_state.attempt_number = 1
        mock_retry_state.fn = "boundary_function"
        mock_retry_state.outcome = "timeout"

        log_attempt_retry(mock_retry_state)

        mock_logger.log.assert_called_once_with(
            logging.WARNING, "Retrying %s: attempt %s ended with: %s", "boundary_function", 1, "timeout"
        )

    @patch("src.utils.logging_helper.logger")
    def test_log_attempt_retry_different_outcomes(self, mock_logger):
        """Test logging with different outcome types."""
        test_cases = [
            (0, "exception_occurred", logging.INFO),
            (3, "network_error", logging.WARNING),
            (1, None, logging.WARNING),
        ]

        for attempt_num, outcome, expected_level in test_cases:
            mock_logger.reset_mock()
            mock_retry_state = Mock(spec=RetryCallState)
            mock_retry_state.attempt_number = attempt_num
            mock_retry_state.fn = f"test_function_{attempt_num}"
            mock_retry_state.outcome = outcome

            log_attempt_retry(mock_retry_state)

            mock_logger.log.assert_called_once_with(
                expected_level,
                "Retrying %s: attempt %s ended with: %s",
                f"test_function_{attempt_num}",
                attempt_num,
                outcome,
            )
