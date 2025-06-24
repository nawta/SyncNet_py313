"""Logging utilities for SyncNet components."""

import logging
import sys
from pathlib import Path
from typing import Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """JSON structured formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class LoggerManager:
    """Centralized logger management."""
    
    _loggers: dict[str, logging.Logger] = {}
    _default_level: int = logging.INFO
    _log_file: Path | None = None
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(cls._default_level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(console_handler)
        
        # File handler if configured
        if cls._log_file:
            file_handler = logging.FileHandler(cls._log_file)
            file_handler.setFormatter(
                StructuredFormatter()
            )
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def configure(
        cls, 
        level: str = "INFO", 
        log_file: Path | str | None = None,
        structured: bool = False
    ) -> None:
        """Configure logging globally.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            structured: Use structured JSON logging for console
        """
        cls._default_level = getattr(logging, level.upper())
        
        if log_file:
            cls._log_file = Path(log_file)
            cls._log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Update existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(cls._default_level)
            
            # Update handlers
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and structured:
                    handler.setFormatter(StructuredFormatter())
    
    @classmethod
    def disable_color(cls) -> None:
        """Disable colored output (useful for CI/CD)."""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(
                        logging.Formatter(
                            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S'
                        )
                    )


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return LoggerManager.get_logger(name)


class LogContext:
    """Context manager for adding extra fields to logs."""
    
    def __init__(self, logger: logging.Logger, **kwargs: Any):
        """Initialize log context.
        
        Args:
            logger: Logger instance
            **kwargs: Extra fields to add to logs
        """
        self.logger = logger
        self.extra_fields = kwargs
        self._old_factory = None
    
    def __enter__(self) -> 'LogContext':
        """Enter context."""
        self._old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = self._old_factory(*args, **kwargs)
            record.extra_fields = self.extra_fields
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit context."""
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, total: int, desc: str = "Processing"):
        """Initialize progress logger.
        
        Args:
            logger: Logger instance
            total: Total number of items
            desc: Description of operation
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        progress = self.current / self.total * 100
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({progress:.1f}%) - ETA: {eta:.1f}s"
            )
    
    def close(self) -> None:
        """Close progress logger."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.desc} completed: {self.total} items in {elapsed:.1f}s "
            f"({self.total / elapsed:.1f} items/s)"
        )


# Convenience functions for module-level logging
def debug(msg: str, **kwargs: Any) -> None:
    """Log debug message."""
    logger = get_logger('syncnet')
    with LogContext(logger, **kwargs):
        logger.debug(msg)


def info(msg: str, **kwargs: Any) -> None:
    """Log info message."""
    logger = get_logger('syncnet')
    with LogContext(logger, **kwargs):
        logger.info(msg)


def warning(msg: str, **kwargs: Any) -> None:
    """Log warning message."""
    logger = get_logger('syncnet')
    with LogContext(logger, **kwargs):
        logger.warning(msg)


def error(msg: str, **kwargs: Any) -> None:
    """Log error message."""
    logger = get_logger('syncnet')
    with LogContext(logger, **kwargs):
        logger.error(msg)


def exception(msg: str, **kwargs: Any) -> None:
    """Log exception with traceback."""
    logger = get_logger('syncnet')
    with LogContext(logger, **kwargs):
        logger.exception(msg)