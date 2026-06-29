from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from emotion_detection_pipeline import main as api


def test_transcribe_requires_one_input() -> None:
    client = TestClient(api.app)

    response = client.post("/transcribe", data={})

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Must provide either a file upload or YouTube URL"
    )


def test_transcribe_rejects_file_and_youtube_url(monkeypatch) -> None:
    client = TestClient(api.app)

    response = client.post(
        "/transcribe",
        data={"youtube_url": "https://example.com/video"},
        files={"file": ("clip.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide either file OR YouTube URL, not both"


def test_transcribe_starts_job_without_running_heavy_work(monkeypatch) -> None:
    async def fake_process_transcription(**kwargs):
        return None

    def fake_create_task(coro):
        coro.close()
        return object()

    monkeypatch.setattr(api, "process_transcription", fake_process_transcription)
    monkeypatch.setattr(api.asyncio, "create_task", fake_create_task)
    api.jobs_storage.clear()

    client = TestClient(api.app)
    response = client.post("/transcribe", data={"youtube_url": "https://example.com"})

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] in api.jobs_storage
    assert body["download_url"] == f"/download/{body['job_id']}"


def test_download_statuses(tmp_path: Path) -> None:
    client = TestClient(api.app)
    api.jobs_storage.clear()

    missing = client.get("/download/missing")
    assert missing.status_code == 404

    api.jobs_storage["processing"] = {
        "status": api.JobStatus.PROCESSING,
        "created_at": datetime.now(),
        "result": None,
        "error": None,
        "artifacts_path": None,
    }
    assert client.get("/download/processing").status_code == 202

    api.jobs_storage["failed"] = {
        "status": api.JobStatus.FAILED,
        "created_at": datetime.now(),
        "result": None,
        "error": "boom",
        "artifacts_path": None,
    }
    failed = client.get("/download/failed")
    assert failed.status_code == 500
    assert "boom" in failed.json()["detail"]

    result_file = tmp_path / "result.txt"
    result_file.write_text("done", encoding="utf-8")
    api.jobs_storage["completed"] = {
        "status": api.JobStatus.COMPLETED,
        "created_at": datetime.now(),
        "result": None,
        "error": None,
        "artifacts_path": str(result_file),
    }

    completed = client.get("/download/completed")
    assert completed.status_code == 200
    assert completed.content == b"done"


def test_process_transcription_marks_failures_for_unsupported_file_type() -> None:
    job_id = "failed-job"
    api.jobs_storage[job_id] = {
        "status": api.JobStatus.PROCESSING,
        "created_at": datetime.now(),
        "result": None,
        "error": None,
        "artifacts_path": None,
    }

    import asyncio

    asyncio.run(
        api.process_transcription(
            job_id=job_id,
            file_data={
                "content": b"bad",
                "filename": "bad.txt",
                "content_type": "text/plain",
            },
            youtube_url=None,
            language="pl",
            engine=api.TranscriptionEngine.WHISPER,
            auto_correction=False,
        )
    )

    assert api.jobs_storage[job_id]["status"] == api.JobStatus.FAILED
    assert "Unsupported file type" in api.jobs_storage[job_id]["error"]


def test_process_transcription_completes_file_job(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    class FakeSpeachToText:
        def __init__(self, **kwargs):
            assert kwargs["preferred_engine"] == "whisper"

        async def transcribe(self, **kwargs):
            return {"artifacts_dir": "job_artifacts"}

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(api, "SpeachToText", FakeSpeachToText)
    monkeypatch.setattr(api.asyncio, "sleep", no_sleep)

    job_id = "completed-job"
    api.jobs_storage[job_id] = {
        "status": api.JobStatus.PROCESSING,
        "created_at": datetime.now(),
        "result": None,
        "error": None,
        "artifacts_path": None,
    }

    import asyncio

    asyncio.run(
        api.process_transcription(
            job_id=job_id,
            file_data={
                "content": b"audio",
                "filename": "clip.wav",
                "content_type": "audio/wav",
            },
            youtube_url=None,
            language="pl",
            engine=api.TranscriptionEngine.WHISPER,
            auto_correction=False,
        )
    )

    assert api.jobs_storage[job_id]["status"] == api.JobStatus.COMPLETED
    assert api.jobs_storage[job_id]["artifacts_path"] == "data/job_artifacts"
