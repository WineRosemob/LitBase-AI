from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOGGING_CONFIGURED = False
_CURRENT_LOG_FILE: Path | None = None


class LoggerFactory:
    """Factory for terminal + file logging configuration."""

    @staticmethod
    def setup(
        output_dir: Path | None = None,
        verbose: bool = False,
        quiet: bool = False,
        force: bool = False,
    ) -> Path | None:
        """Configure logging handlers for console and optional run.log file."""
        global _LOGGING_CONFIGURED, _CURRENT_LOG_FILE
        if _LOGGING_CONFIGURED and not force:
            return _CURRENT_LOG_FILE

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        for handler in list(root.handlers):
            root.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console_handler)

        # Reduce noisy third-party request logs in normal mode.
        noisy_loggers = [
            "httpx",
            "httpcore",
            "urllib3",
            "playwright",
            "asyncio",
        ]
        noisy_level = logging.INFO if verbose else logging.WARNING
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(noisy_level)

        log_file_path: Path | None = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = output_dir / "run.log"
            file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            root.addHandler(file_handler)

        _CURRENT_LOG_FILE = log_file_path
        _LOGGING_CONFIGURED = True
        return log_file_path


def setup_logger(level: str = "INFO") -> None:
    """Compatibility wrapper for legacy imports."""
    verbose = level.upper() == "DEBUG"
    LoggerFactory.setup(verbose=verbose, quiet=False, force=False)


def get_logger(name: str) -> logging.Logger:
    """Return configured logger by name."""
    if not _LOGGING_CONFIGURED:
        LoggerFactory.setup()
    return logging.getLogger(name)
