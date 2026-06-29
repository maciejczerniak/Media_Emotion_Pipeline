from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from emotion_detection_pipeline.speach_to_text.speech_to_text_assemblyAI import (
    AssemblyAITranscriber,
    _create_transcription_dataframe as create_assembly_dataframe,
)
from emotion_detection_pipeline.speach_to_text.speech_to_text_whisper import (
    WhisperTranscriber,
    _create_transcription_dataframe as create_whisper_dataframe,
    _format_timestamp,
)
from emotion_detection_pipeline.speach_to_text.utils.save_transcription_artifacts import (
    _save_artifacts,
)


def test_format_timestamp_uses_minutes_and_seconds() -> None:
    assert _format_timestamp(125.9) == "02:05"


def test_whisper_dataframe_from_segments() -> None:
    df = create_whisper_dataframe(
        {
            "segments": [
                {
                    "id": 7,
                    "start": 1.234,
                    "end": 4.789,
                    "text": " Hello ",
                    "no_speech_prob": 0.2,
                    "avg_logprob": -0.1,
                }
            ]
        }
    )

    assert df.to_dict("records") == [
        {
            "Segment_id": 7,
            "Start_time_seconds": 1.23,
            "End_time_seconds": 4.79,
            "Start_time_formatted": "00:01",
            "End_time_formatted": "00:04",
            "Duration_seconds": 3.55,
            "Sentence": "Hello",
            "No_speech_prob": 0.2,
            "Avg_logprob": -0.1,
        }
    ]


def test_assembly_dataframe_uses_segments_when_available() -> None:
    transcript = SimpleNamespace(
        segments=[SimpleNamespace(start=1000, end=2500, text=" Segment text ")]
    )

    df = create_assembly_dataframe(transcript)

    assert df[["Start_time_seconds", "End_time_seconds", "Sentence"]].to_dict(
        "records"
    ) == [
        {
            "Start_time_seconds": 1.0,
            "End_time_seconds": 2.5,
            "Sentence": "Segment text",
        }
    ]


def test_assembly_dataframe_can_build_sentences_from_words() -> None:
    transcript = SimpleNamespace(
        segments=None,
        words=[
            SimpleNamespace(start=0, end=300, text="Hello"),
            SimpleNamespace(start=300, end=700, text="there."),
            SimpleNamespace(start=900, end=1200, text="Again"),
        ],
    )

    df = create_assembly_dataframe(transcript)

    assert df["Sentence"].tolist() == ["Hello there.", "Again"]


def test_assembly_dataframe_requires_segments_or_words() -> None:
    with pytest.raises(RuntimeError, match="neither 'segments' nor 'words'"):
        create_assembly_dataframe(SimpleNamespace(segments=None, words=None))


def test_save_artifacts_writes_text_and_segments(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.wav"
    segments_df = pd.DataFrame({"Sentence": ["hello"]})

    artifacts_dir = _save_artifacts(
        video_path,
        {"full_text": "hello world", "segments_df": segments_df},
    )

    assert artifacts_dir == tmp_path / "clip_artifacts"
    assert (artifacts_dir / "clip_transcription.txt").read_text(
        encoding="utf-8"
    ) == "hello world"
    assert (artifacts_dir / "clip_segments.csv").exists()


def test_whisper_transcriber_can_return_text_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "emotion_detection_pipeline.speach_to_text.speech_to_text_whisper.whisper.load_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "emotion_detection_pipeline.speach_to_text.speech_to_text_whisper.whisper.transcribe",
        lambda *args, **kwargs: {"text": "hello", "segments": []},
    )

    transcriber = WhisperTranscriber(model_size="tiny")

    assert (
        transcriber.transcribe(tmp_path / "clip.wav", return_only_text=True) == "hello"
    )


def test_whisper_transcriber_can_return_artifacts_without_saving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "emotion_detection_pipeline.speach_to_text.speech_to_text_whisper.whisper.load_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "emotion_detection_pipeline.speach_to_text.speech_to_text_whisper.whisper.transcribe",
        lambda *args, **kwargs: {
            "text": "hello",
            "segments": [{"id": 1, "start": 0, "end": 1, "text": "hello"}],
        },
    )

    transcriber = WhisperTranscriber(model_size="tiny")
    result = transcriber.transcribe(tmp_path / "clip.wav", save_artifacts=False)

    assert result["full_text"] == "hello"
    assert result["segments_df"]["Sentence"].tolist() == ["hello"]


def test_assembly_transcriber_can_return_text_only(monkeypatch, tmp_path: Path) -> None:
    class FakeConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        "emotion_detection_pipeline.speach_to_text.speech_to_text_assemblyAI.aai.TranscriptionConfig",
        FakeConfig,
    )

    transcriber = AssemblyAITranscriber(api_key="key")
    monkeypatch.setattr(
        transcriber,
        "_generate_transcription",
        lambda video_path: SimpleNamespace(text="hello"),
    )

    assert (
        transcriber.transcribe(tmp_path / "clip.wav", return_only_text=True) == "hello"
    )
