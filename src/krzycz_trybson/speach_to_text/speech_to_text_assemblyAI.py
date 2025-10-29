# Transcribe audio file using AssemblyAI and save the transcript with timestamps to an Excel file
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import pandas as pd
from typing_extensions import Literal

from krzycz_trybson.config import settings
from krzycz_trybson.speach_to_text.utils.save_transcription_artifacts import (
    _save_artifacts,
)
import assemblyai as aai  # type: ignore[import-untyped]


def _format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def _create_transcription_dataframe(assembly_result: aai.Transcript) -> pd.DataFrame:
    """Convert Whisper result to DataFrame with segments and timestamps"""

    segments_data: List[Dict[str, Any]] = []

    # Path A: use native segments if available
    if getattr(assembly_result, "segments", None):
        for i, seg in enumerate(assembly_result.segments):
            start_s = round((seg.start or 0) / 1000.0, 2)
            end_s = round((seg.end or 0) / 1000.0, 2)
            duration_s = round(end_s - start_s, 2)
            segments_data.append(
                {
                    "Segment_id": i,
                    "Start_time_seconds": start_s,
                    "End_time_seconds": end_s,
                    "Start_time_formatted": _format_timestamp(start_s),
                    "End_time_formatted": _format_timestamp(end_s),
                    "Duration_seconds": duration_s,
                    "Sentence": (seg.text or "").strip(),
                }
            )

    # Path B: synthesize segments from words
    elif getattr(assembly_result, "words", None):

        def ms_to_sec(ms: Optional[int]) -> float:
            return round(((ms or 0) / 1000.0), 2)

        sentence_tokens: List[str] = []
        start_ms: Optional[int] = None
        end_ms: Optional[int] = None
        seg_id = 0

        for w in assembly_result.words:
            if start_ms is None:
                start_ms = w.start
            sentence_tokens.append(w.text)
            end_ms = w.end

            # Sentence boundary: ., ?, !
            if w.text and w.text[-1] in ".?!":
                start_s = ms_to_sec(start_ms)
                end_s = ms_to_sec(end_ms)
                duration_s = round(end_s - start_s, 2)
                segments_data.append(
                    {
                        "Segment_id": seg_id,
                        "Start_time_seconds": start_s,
                        "End_time_seconds": end_s,
                        "Start_time_formatted": _format_timestamp(start_s),
                        "End_time_formatted": _format_timestamp(end_s),
                        "Duration_seconds": duration_s,
                        "Sentence": " ".join(sentence_tokens).strip(),
                    }
                )
                seg_id += 1
                sentence_tokens = []
                start_ms = None
                end_ms = None

        # Flush trailing words (if the audio ends mid-sentence)
        if sentence_tokens and end_ms is not None:
            start_s = ms_to_sec(start_ms)
            end_s = ms_to_sec(end_ms)
            duration_s = round(end_s - start_s, 2)
            segments_data.append(
                {
                    "Segment_id": seg_id,
                    "Start_time_seconds": start_s,
                    "End_time_seconds": end_s,
                    "Start_time_formatted": _format_timestamp(start_s),
                    "End_time_formatted": _format_timestamp(end_s),
                    "Duration_seconds": duration_s,
                    "Sentence": " ".join(sentence_tokens).strip(),
                }
            )
    else:
        raise RuntimeError("AssemblyAI transcript has neither 'segments' nor 'words'.")

    df = pd.DataFrame(segments_data)
    return df


class AssemblyAITranscriber:
    def __init__(
        self,
        model: Literal["universal"] = "universal",
        language: str = "pl",
        api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: AssemblyAI model name.
            api_key: If None, will read from env var ASSEMBLYAI_API_KEY.
            language: Language code (default 'pl').
        """
        self.model = model
        self.api_key = api_key or settings.assemblyai_api_key
        self.language = language

        if not self.api_key:
            raise RuntimeError("Missing AssemblyAI API key.")

        aai.settings.api_key = self.api_key
        self._config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal,
            language_code=self.language,
            speaker_labels=True,
        )

    def _generate_transcription(
        self, video_path: Path
    ) -> Any:  # Changed from Dict[str, Any] to Any since assemblyai returns untyped
        """
        Generate transcription using AssemblyAI model.
        Args:
            video_path: Path to the video/audio file.

        Returns a dictionary with transcription results.

        """
        transcriber = aai.Transcriber(config=self._config)
        transcript = transcriber.transcribe(str(video_path))
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        return transcript

    def transcribe(
        self,
        video_path: Path,
        return_only_text: bool = False,
        save_artifacts: bool = True,
    ) -> Union[str, Dict[str, Union[str, pd.DataFrame]]]:
        """
        Generate transcription using AssemblyAI model.
        Args:
            video_path: Path to the video/audio file.
            return_only_text: If True,
            return only the transcribed text. If False, return full result with segments DataFrame.
            save_artifacts: If True, save the transcription artifacts to files.
        Returns: Dictionary with either full transcription text or full result including segments DataFrame.
        """

        result = self._generate_transcription(video_path)
        if return_only_text:
            return str(result.text)  # Explicit cast to str

        df = _create_transcription_dataframe(result)
        result = {"full_text": result.text or "", "segments_df": df}

        if save_artifacts:
            artifacts_dir = _save_artifacts(video_path, result)
            result["artifacts_dir"] = str(artifacts_dir)

        return result
