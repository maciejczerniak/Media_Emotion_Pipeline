import logging
from pathlib import Path
from typing import Optional

from media_emotion_pipeline.config import Settings, settings


def get_logger(
    name: Optional[str] = None, settings: Settings = settings
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.get_log_level())

    logger.handlers.clear()

    logger.addHandler(get_log_file_handler(settings=settings))

    if settings.get_environment_info().get("is_development", False):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(settings.get_log_level())
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_log_file_handler(settings: Settings = settings) -> logging.FileHandler:
    raw_path = settings.get_logging_config().get("path")
    if raw_path is None:
        raise ValueError("log_path must be defined")

    log_path = Path(raw_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path)
    handler.setLevel(settings.get_log_level())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    return handler
