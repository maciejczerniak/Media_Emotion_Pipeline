from pathlib import Path
from typing import Literal, Dict, Any, Union
from emotion_detection_pipeline.logger import get_logger
from emotion_detection_pipeline.speach_to_text.speech_to_text_assemblyAI import (
    AssemblyAITranscriber,
)
from emotion_detection_pipeline.speach_to_text.speech_to_text_whisper import (
    WhisperTranscriber,
)

logger = get_logger(__name__)


class SpeachToText:
    def __init__(
        self,
        preferred_engine: Literal["whisper", "assemblyai"],
        auto_correction: bool,
        language: str = "pl",
    ) -> None:
        """
        Initialize the SpeachToText with the preferred STT engine.
        Args:
            preferred_engine: The STT engine to use ('whisper' or 'assemblyai').
            auto_correction: Whether to apply auto-correction to the transcribed text.
            language: Language code for transcription (default is 'pl' for Polish).
        """
        self.preferred_engine = preferred_engine
        self.auto_correction = auto_correction
        self.language = language

    async def transcribe(
        self,
        video_path: Path,
        language: str = "pl",
        return_only_text: bool = False,
        save_artifacts: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe the given video/audio file using the preferred STT engine.
        Args:
            save_artifacts: If True, save the transcription artifacts to files.
            video_path: Path to the video/audio file.
            language:  Language code for transcription (default is 'pl' for Polish).
            return_only_text:  If True, return only the transcribed text. If False, return full result with segments DataFrame.

        Returns: Dictionary with either full transcription text or full result including segments DataFrame.
        """
        result: Dict[str, Any] = {}
        transcriber: Union[WhisperTranscriber, AssemblyAITranscriber]

        if self.preferred_engine == "whisper":
            transcriber = WhisperTranscriber()
            transcriber_result = transcriber.transcribe(
                video_path,
                language=language,
                return_only_text=return_only_text,
                save_artifacts=save_artifacts,
            )

            if return_only_text:
                result["full_text"] = str(transcriber_result)
            else:
                result = dict(transcriber_result)  # type: ignore

        elif self.preferred_engine == "assemblyai":
            transcriber = AssemblyAITranscriber()
            transcriber_result = transcriber.transcribe(
                video_path,
                return_only_text=return_only_text,
                save_artifacts=save_artifacts,
            )
            if return_only_text:
                result = {"full_text": str(transcriber_result)}
            else:
                result = dict(transcriber_result)  # type: ignore

        if self.auto_correction and not return_only_text:
            raise NotImplementedError

        return result
