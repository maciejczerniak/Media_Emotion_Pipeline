from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import whisper  # type: ignore[import-untyped]
from typing_extensions import Literal

from krzycz_trybson.speach_to_text.utils.save_transcription_artifacts import (
    _save_artifacts,
)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _create_transcription_dataframe(whisper_result: Dict[str, Any]) -> pd.DataFrame:
    """Convert Whisper result to DataFrame with segments and timestamps"""

    segments_data = []

    for segment in whisper_result["segments"]:
        segments_data.append(
            {
                "Segment_id": segment["id"],
                "Start_time_seconds": round(segment["start"], 2),
                "End_time_seconds": round(segment["end"], 2),
                "Start_time_formatted": _format_timestamp(segment["start"]),
                "End_time_formatted": _format_timestamp(segment["end"]),
                "Duration_seconds": round(segment["end"] - segment["start"], 2),
                "Sentence": segment["text"].strip(),
                "No_speech_prob": segment.get("no_speech_prob", 0.0),
                "Avg_logprob": segment.get("avg_logprob", 0.0),
            }
        )

    df = pd.DataFrame(segments_data)
    return df


class WhisperTranscriber:
    def __init__(
        self,
        model_size: Literal[
            "tiny", "base", "small", "medium", "large", "turbo"
        ] = "large",
    ) -> None:
        self.model_size = model_size
        self.model = whisper.load_model(model_size, device="cuda")

    def _generate_transcription(
        self, video_path: Path, language: str = "pl"
    ) -> Any:  # Changed from Dict[str, Any] to Any since whisper returns untyped
        """
        Generate transcription using Whisper model.
        Args:
            video_path: Path to the video/audio file.
            language: Language code for transcription (default is 'pl' for Polish).

        Returns a dictionary with transcription results.

        """
        result = whisper.transcribe(
            self.model,
            str(video_path),
            language=language,
        )
        return result

    def transcribe(
        self,
        video_path: Path,
        language: str = "pl",
        return_only_text: bool = False,
        save_artifacts: bool = True,
    ) -> Union[str, Dict[str, Union[str, pd.DataFrame]]]:
        """
        Generate transcription using Whisper model.
        Args:
            video_path: Path to the video/audio file.
            language: Language code for transcription (default is 'pl' for Polish).
            return_only_text: If True, return only the transcribed text. If False, return full result with segments DataFrame.
            save_artifacts: If True, save the transcription artifacts to files.

        Returns: Dictionary with either full transcription text or full result including segments DataFrame.
        """

        result = self._generate_transcription(video_path, language)
        if return_only_text:
            return str(result["text"])  # Explicit cast to str

        df = _create_transcription_dataframe(result)
        result = {"full_text": result["text"], "segments_df": df}

        if save_artifacts:
            artifacts_dir = _save_artifacts(video_path, result)
            result["artifacts_dir"] = str(artifacts_dir)

        return result
