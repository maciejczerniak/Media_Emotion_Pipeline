import asyncio
from pathlib import Path

import pandas as pd

from emotion_detection_pipeline import pipeline as pipeline_module


def test_parse_args_for_local_file(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--input", "video.mp4", "--output", "results", "--no-autocorrection"],
    )

    args = pipeline_module.parse_args()

    assert args.input == "video.mp4"
    assert args.output == "results"
    assert args.autocorrection is False


def test_pipeline_orchestrates_without_autocorrection(
    monkeypatch, tmp_path: Path
) -> None:
    artifacts_dir = tmp_path / "clip_artifacts"
    artifacts_dir.mkdir()
    pd.DataFrame({"Sentence": ["czesc"]}).to_csv(
        artifacts_dir / "clip_segments.csv",
        index=False,
    )

    class FakeSpeachToText:
        def __init__(self, **kwargs):
            pass

        async def transcribe(self, **kwargs):
            return {"artifacts_dir": str(artifacts_dir)}

    class FakePredictor:
        def predict_dataframe(self, df, text_column, include_confidence):
            df["Emotion_core"] = ["joy"]
            df["confidence"] = [0.9]
            return df

    monkeypatch.setattr(pipeline_module, "SpeachToText", FakeSpeachToText)
    monkeypatch.setattr(pipeline_module, "translate_batch", lambda texts: ["hello"])
    monkeypatch.setattr(pipeline_module, "EmotionPredictor", lambda: FakePredictor())

    output_dir = tmp_path / "results"
    output_dir.mkdir()

    df = asyncio.run(
        pipeline_module.pipeline(
            file_path=tmp_path / "clip.wav",
            output_path=output_dir,
            autocorrection=False,
        )
    )

    assert df["Translation"].tolist() == ["hello"]
    assert df["Emotion_core"].tolist() == ["joy"]
    assert (output_dir / "clip_processed.csv").exists()


def test_pipeline_orchestrates_with_autocorrection(monkeypatch, tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "clip_artifacts"
    artifacts_dir.mkdir()
    pd.DataFrame({"Sentence": ["czesc"]}).to_csv(
        artifacts_dir / "clip_segments.csv",
        index=False,
    )

    class FakeSpeachToText:
        def __init__(self, **kwargs):
            pass

        async def transcribe(self, **kwargs):
            return {"artifacts_dir": str(artifacts_dir)}

    class FakePredictor:
        def predict_dataframe(self, df, text_column, include_confidence):
            assert text_column == "Translation"
            df["Emotion_core"] = ["joy"]
            return df

    monkeypatch.setattr(pipeline_module, "SpeachToText", FakeSpeachToText)
    monkeypatch.setattr(pipeline_module.ltp, "LanguageTool", lambda language: object())
    monkeypatch.setattr(
        pipeline_module,
        "auto_correct_batch",
        lambda df, *args, **kwargs: df.assign(Sentence_corrected=["poprawione"]),
    )
    monkeypatch.setattr(pipeline_module, "translate_batch", lambda texts: ["corrected"])
    monkeypatch.setattr(pipeline_module, "EmotionPredictor", lambda: FakePredictor())

    output_dir = tmp_path / "results"
    output_dir.mkdir()

    df = asyncio.run(
        pipeline_module.pipeline(
            file_path=tmp_path / "clip.wav",
            output_path=output_dir,
            autocorrection=True,
            url_to_llm="http://ollama",
        )
    )

    assert df["Sentence_corrected"].tolist() == ["poprawione"]
    assert df["Translation"].tolist() == ["corrected"]


def test_main_processes_local_file(monkeypatch, tmp_path: Path) -> None:
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"audio")
    output_dir = tmp_path / "results"
    calls = {}

    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "input": str(input_file),
                "youtube_url": None,
                "output": str(output_dir),
                "autocorrection": False,
                "llm_url": "http://ollama",
                "save_dir": str(tmp_path / "data"),
            },
        )(),
    )

    async def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"ok": [True]})

    monkeypatch.setattr(pipeline_module, "pipeline", fake_pipeline)

    result = asyncio.run(pipeline_module.main())

    assert result["ok"].tolist() == [True]
    assert calls["file_path"] == input_file
    assert output_dir.exists()


def test_main_raises_for_missing_local_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "input": str(tmp_path / "missing.wav"),
                "youtube_url": None,
                "output": str(tmp_path / "results"),
                "autocorrection": False,
                "llm_url": "http://ollama",
                "save_dir": str(tmp_path / "data"),
            },
        )(),
    )

    import pytest

    with pytest.raises(FileNotFoundError):
        asyncio.run(pipeline_module.main())
