"""Tests for logging utilities."""

import logging
import time
from pathlib import Path

import pytest

from src.utils.logging import LoggerMixin, get_logger, log_execution_time, setup_logger


class TestLoggingSetup:
    """Tests for logger setup functions."""

    def test_get_logger(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_get_logger_with_file(self, tmp_path):
        """Test logger creation with file output."""
        log_file = tmp_path / "test.log"
        logger = get_logger("test_logger", log_file=log_file)

        # Verify file handler was added
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].baseFilename == str(log_file)

    def test_get_logger_level(self):
        """Test logger level setting."""
        # Test with integer level
        logger = get_logger("test_logger", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        # Test with string level
        logger = get_logger("test_logger2", level="INFO")
        assert logger.level == logging.INFO

    def test_setup_logger_levels(self):
        """Test logger level configuration."""
        logger = setup_logger("test_logger", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

        logger = setup_logger("test_logger2", level="INFO")
        assert logger.level == logging.INFO


class TestLoggerMixin:
    """Tests for LoggerMixin class."""

    class SampleClass(LoggerMixin):
        """Sample class using LoggerMixin."""

        pass

    def test_mixin_logger_creation(self):
        """Test logger creation in mixin."""
        instance = self.SampleClass()
        assert hasattr(instance.__class__, "logger")
        assert isinstance(instance.__class__.logger, logging.Logger)
        assert instance.__class__.logger.name == "SampleClass"

    def test_mixin_multiple_classes(self):
        """Test logger creation with multiple mixin classes."""

        class Class1(LoggerMixin):
            pass

        class Class2(LoggerMixin):
            pass

        assert Class1.logger.name == "Class1"
        assert Class2.logger.name == "Class2"
        assert Class1.logger != Class2.logger


class TestExecutionTimeDecorator:
    """Tests for execution time logging decorator."""

    def test_successful_execution(self, caplog):
        """Test logging of successful execution."""

        @log_execution_time()
        def test_function():
            time.sleep(0.1)
            return "success"

        with caplog.at_level(logging.DEBUG):
            result = test_function()

        assert result == "success"
        assert "executed in" in caplog.text
        assert "seconds" in caplog.text

    def test_failed_execution(self, caplog):
        """Test logging of failed execution."""

        @log_execution_time()
        def failing_function():
            time.sleep(0.1)
            raise ValueError("Test error")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                failing_function()

        assert "failed after" in caplog.text
        assert "Test error" in caplog.text

    def test_custom_message(self, caplog):
        """Test custom message in time logging."""
        custom_msg = "Custom execution time for %s: %.3f seconds"

        @log_execution_time(message=custom_msg)
        def test_function():
            time.sleep(0.1)

        with caplog.at_level(logging.DEBUG):
            test_function()

        assert "Custom execution time" in caplog.text
        assert "seconds" in caplog.text
