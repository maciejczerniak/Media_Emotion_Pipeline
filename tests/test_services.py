from types import SimpleNamespace

import pandas as pd

from media_emotion_pipeline.autocorrection import autocorrection
from media_emotion_pipeline.speach_to_text.utils import download_youtube_video
from media_emotion_pipeline.speach_to_text.stt_master import SpeachToText
from media_emotion_pipeline.translation import translate


def test_correct_with_ollama_returns_json_correction(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self):
            return {"response": '{"corrected": "Poprawiony tekst"}'}

    monkeypatch.setattr(
        autocorrection.requests, "post", lambda *args, **kwargs: Response()
    )

    assert (
        autocorrection.correct_with_ollama("tekst", "http://ollama", "model")
        == "Poprawiony tekst"
    )


def test_correct_with_ollama_falls_back_on_bad_json(monkeypatch) -> None:
    class Response:
        status_code = 200

        def json(self):
            return {"response": "not json"}

    monkeypatch.setattr(
        autocorrection.requests, "post", lambda *args, **kwargs: Response()
    )

    assert (
        autocorrection.correct_with_ollama("original", "http://ollama", "model")
        == "original"
    )


def test_correct_with_ollama_falls_back_on_request_error(monkeypatch) -> None:
    def raise_error(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(autocorrection.requests, "post", raise_error)

    assert (
        autocorrection.correct_with_ollama("original", "http://ollama", "model")
        == "original"
    )


def test_orchestrate_corrections_runs_both_stages(monkeypatch) -> None:
    monkeypatch.setattr(
        autocorrection,
        "correct_with_language_tools",
        lambda text, tool: f"{text} language",
    )
    monkeypatch.setattr(
        autocorrection,
        "correct_with_ollama",
        lambda text, url, model: f"{text} ollama",
    )

    assert (
        autocorrection.orchestrate_corrections(
            "text",
            tool=object(),
            ollama_api_url="http://ollama",
            model="model",
        )
        == "text language ollama"
    )


def test_auto_correct_batch_adds_corrected_text(monkeypatch) -> None:
    monkeypatch.setattr(
        autocorrection,
        "correct_with_language_tools",
        lambda text, tool: f"{text} language",
    )
    monkeypatch.setattr(
        autocorrection,
        "correct_with_ollama",
        lambda text, url, model: f"{text} ollama",
    )

    df = autocorrection.auto_correct_batch(
        pd.DataFrame({"Sentence": ["one", "two"]}),
        "Sentence",
        tool=object(),
        ollama_api_url="http://ollama",
        model="model",
        max_workers=1,
    )

    assert df["corrected_text"].tolist() == [
        "one language ollama",
        "two language ollama",
    ]


def test_translate_batch_uses_model_once_for_batches(monkeypatch) -> None:
    class FakeTensor:
        def to(self, device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def __call__(self, batch, **kwargs):
            return {"input_ids": FakeTensor()}

        def decode(self, output, skip_special_tokens=True):
            return f"translated-{output}"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(self, **inputs):
            return ["a", "b"]

    monkeypatch.setattr(translate, "MarianTokenizer", FakeTokenizer)
    monkeypatch.setattr(translate, "MarianMTModel", FakeModel)
    monkeypatch.setattr(translate.torch.cuda, "is_available", lambda: False)

    assert translate.translate_batch(["jeden", "dwa"], batch_size=2) == [
        "translated-a",
        "translated-b",
    ]


def test_download_sync_returns_title(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def extract_info(self, url, download):
            return {"title": "Video title"}

    monkeypatch.setattr(download_youtube_video, "YoutubeDL", FakeYoutubeDL)

    assert (
        download_youtube_video._download_sync("https://example.com", {})
        == "Video title"
    )


def test_download_sync_returns_uuid_when_title_missing(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        def extract_info(self, url, download):
            return {}

    monkeypatch.setattr(download_youtube_video, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(download_youtube_video.uuid, "uuid4", lambda: "uuid-title")

    assert (
        download_youtube_video._download_sync("https://example.com", {}) == "uuid-title"
    )


def test_fallback_download_returns_uuid_after_failure(monkeypatch) -> None:
    async def fake_run(*args, **kwargs):
        raise RuntimeError("failed")

    class FakeLoop:
        async def run_in_executor(self, *args, **kwargs):
            raise RuntimeError("failed")

    monkeypatch.setattr(
        download_youtube_video.asyncio, "get_event_loop", lambda: FakeLoop()
    )
    monkeypatch.setattr(download_youtube_video.uuid, "uuid4", lambda: "fallback-id")

    import asyncio

    assert (
        asyncio.run(
            download_youtube_video._fallback_download(
                "https://example.com",
                "clip.wav",
            )
        )
        == "fallback-id"
    )


async def _fake_transcribe_result(*args, **kwargs):
    return {"full_text": "hello"}


def test_speach_to_text_routes_to_whisper(monkeypatch):
    created = {}

    class FakeWhisper:
        def transcribe(self, *args, **kwargs):
            created["kwargs"] = kwargs
            return {"full_text": "from whisper"}

    monkeypatch.setattr(
        "media_emotion_pipeline.speach_to_text.stt_master.WhisperTranscriber",
        lambda: FakeWhisper(),
    )

    service = SpeachToText(preferred_engine="whisper", auto_correction=False)

    import asyncio

    result = asyncio.run(
        service.transcribe(video_path=SimpleNamespace(), language="pl")
    )

    assert result == {"full_text": "from whisper"}
    assert created["kwargs"]["language"] == "pl"


def test_speach_to_text_routes_to_assemblyai(monkeypatch):
    class FakeAssembly:
        def transcribe(self, *args, **kwargs):
            return {"full_text": "from assembly"}

    monkeypatch.setattr(
        "media_emotion_pipeline.speach_to_text.stt_master.AssemblyAITranscriber",
        lambda: FakeAssembly(),
    )

    service = SpeachToText(preferred_engine="assemblyai", auto_correction=False)

    import asyncio

    result = asyncio.run(service.transcribe(video_path=SimpleNamespace()))

    assert result == {"full_text": "from assembly"}
