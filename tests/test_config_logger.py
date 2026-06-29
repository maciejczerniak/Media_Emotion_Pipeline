import logging
from pathlib import Path

from emotion_detection_pipeline.config import Settings
from emotion_detection_pipeline.logger import get_logger


def test_settings_defaults_use_project_name() -> None:
    settings = Settings()

    assert settings.app_name == "emotion-detection-pipeline"
    assert settings.get_app_info()["name"] == "emotion-detection-pipeline"
    assert settings.get_log_level() == logging.INFO


def test_version_components_are_parsed() -> None:
    settings = Settings(version="1.2.3-rc1+build5")

    assert settings.get_version_components() == {
        "major": "1",
        "minor": "2",
        "patch": "3",
        "prerelease": "rc1",
        "build": "build5",
    }


def test_logger_creates_configured_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    settings = Settings(log_file=log_path, status="production")

    logger = get_logger("test_logger_creates_configured_log_file", settings=settings)
    logger.info("hello")

    for handler in logger.handlers:
        handler.flush()
        handler.close()

    assert log_path.exists()
    assert "hello" in log_path.read_text(encoding="utf-8")
