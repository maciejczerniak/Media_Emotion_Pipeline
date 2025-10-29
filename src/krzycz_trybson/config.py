import re
import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    # Application settings
    app_name: str = Field(default="krzycz-trybson", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    authors: List[str] = Field(default=[""], description="List of authors")
    status: Literal["development", "production"] = Field(
        default="development", description="Application status"
    )

    # Debug settings
    debug: bool = Field(default=False, description="Enable debug mode")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(
        default=DOTENV.parent / "logs" / "krzycz-trybson.log",
        description="Log file path",
    )

    # API settings
    api_port: int = Field(default=8000, description="API port")

    # External services settings
    assemblyai_api_key: str = Field(
        default="7f70532ec0ed489a9eb676e7048021c2", description="AssemblyAI API key"
    )

    # Validators
    @field_validator("debug")
    @classmethod
    def set_and_verify_debug_mode(cls, v: Optional[bool], info: ValidationInfo) -> bool:
        status = info.data.get("status")
        if v is None:
            return status == "DEVELOPMENT"
        if status == "PRODUCTION" and v:
            raise ValueError("Debug mode cannot be enabled in PRODUCTION environment")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v is None:
            return "DEBUG" if info.data.get("debug", False) else "INFO"

        valid_levels = [
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARNING",
            "WARN",
            "INFO",
            "DEBUG",
            "NOTSET",
        ]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
            )
        return v

    @field_validator("assemblyai_api_key")
    @classmethod
    def validate_assemblyai_api_key(cls, v: Optional[str], info: ValidationInfo) -> str:
        if not v or v.strip() == "":
            raise ValueError("AssemblyAI API key is required")
        return v

    model_config = SettingsConfigDict(
        env_file=DOTENV,
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Getter methods
    def get_debug_state(self) -> bool:
        """Get the debug state as a boolean."""
        return bool(self.debug)

    def get_log_level(self) -> int:
        """Convert the string log level to the corresponding integer value."""
        level_map: Dict[str, int] = {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        # Ensure log_level is not None before using as dict key
        log_level = self.log_level
        return level_map[log_level]

    def get_app_info(self) -> Dict[str, Any]:
        """Get a dictionary with basic application information."""
        return {
            "name": self.app_name,
            "version": self.version,
            "environment": self.status,
            "debug": self.debug,
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get a dictionary with logging configuration."""
        return {
            "level": self.log_level,
            "level_int": self.get_log_level(),
            "path": str(self.log_file),
            "directory": str(self.log_file.parent),
        }

    def is_development(self) -> bool:
        """Check if the application is running in development mode."""
        return self.status == "development"

    def is_production(self) -> bool:
        """Check if the application is running in production mode."""
        return self.status == "production"

    def get_version_components(self) -> Dict[str, str]:
        """Parse and return the components of the semantic version."""
        # Parse version like 1.0.0-rc2+build123
        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$",
            self.version,
        )
        if match:
            major, minor, patch, prerelease, build = match.groups()
            return {
                "major": major,
                "minor": minor,
                "patch": patch,
                "prerelease": prerelease or "",
                "build": build or "",
            }
        return {"major": "", "minor": "", "patch": "", "prerelease": "", "build": ""}

    def as_dict(self) -> Dict[str, Any]:
        """Return all settings as a dictionary."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "authors": self.authors,
            "status": self.status,
            "debug": self.debug,
            "log_level": self.log_level,
            "log_path": str(self.log_file),
        }

    def get_environment_info(self) -> dict[str, str | bool | None | Any]:
        """Get information about the current environment."""
        return {
            "status": self.status,
            "debug": self.debug,
            "is_development": self.is_development(),
            "is_production": self.is_production(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        }

    def get_assemblyai_api_key(self) -> str:
        return self.assemblyai_api_key


def get_settings(use_test_env: bool = False) -> Settings:
    """Get settings instance with appropriate environment configuration."""
    if use_test_env:
        base_dir = Path(__file__).parent.absolute()
        env_test_path = base_dir / ".env.test"

        # Create a temporary settings class with different env file
        class TestSettings(Settings):
            model_config = SettingsConfigDict(
                env_file=str(env_test_path),
                env_file_encoding="utf-8",
                case_sensitive=False,
            )

        return TestSettings()
    return Settings()


settings = get_settings()
