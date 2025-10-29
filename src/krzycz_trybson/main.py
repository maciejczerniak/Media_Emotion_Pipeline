import asyncio
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, BackgroundTasks, UploadFile, Form, HTTPException
from fastapi.params import File
from pydantic import BaseModel
from starlette.responses import FileResponse, JSONResponse

from krzycz_trybson.logger import get_logger
from krzycz_trybson.speach_to_text.stt_master import SpeachToText
from krzycz_trybson.speach_to_text.utils.download_youtube_video import (
    download_youtube_video,
)

logger = get_logger(__name__)
app = FastAPI()

# In-memory job storage (use Redis/database in production)
jobs_storage: Dict[str, Dict[str, Any]] = {}


class TranscriptionEngine(str, Enum):
    WHISPER = "whisper"
    ASSEMBLYAI = "assemblyai"


class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionResponse(BaseModel):
    job_id: str
    download_url: str
    message: str
    estimated_time: str


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    background_tasks: BackgroundTasks,
    # File upload option
    file: Optional[UploadFile] = File(default=None),  # type: ignore
    # YouTube URL option
    youtube_url: Optional[str] = Form(default=None),
    language: str = Form(default="pl"),
    engine: TranscriptionEngine = Form(default=TranscriptionEngine.WHISPER),
    auto_correction: bool = Form(default=False),
) -> TranscriptionResponse:
    """
    Start a transcription job for an audio file or YouTube video.
    Returns job_id and download_url immediately.
    """

    # Validate input - must provide either file or URL
    if not file and not youtube_url:
        raise HTTPException(
            status_code=400, detail="Must provide either a file upload or YouTube URL"
        )

    if file and youtube_url:
        raise HTTPException(
            status_code=400, detail="Provide either file OR YouTube URL, not both"
        )

    job_id = str(uuid.uuid4())
    download_url = f"/download/{job_id}"

    # Initialize job in storage
    jobs_storage[job_id] = {
        "status": JobStatus.PROCESSING,
        "created_at": datetime.now(),
        "result": None,
        "error": None,
        "artifacts_path": None,
    }

    # Prepare file data if file upload
    file_data = None
    if file:
        content = await file.read()
        file_data = {
            "content": content,
            "filename": file.filename,
            "content_type": file.content_type,
        }

    # Start background processing immediately
    asyncio.create_task(
        process_transcription(
            job_id=job_id,
            file_data=file_data,
            youtube_url=youtube_url,
            language=language,
            engine=engine,
            auto_correction=auto_correction,
        )
    )

    estimated_time = "2-5 minutes for audio files, 5-15 minutes for YouTube videos"

    return TranscriptionResponse(
        job_id=job_id,
        download_url=download_url,
        message="Transcription started. Your download will be ready at the provided URL.",
        estimated_time=estimated_time,
    )


@app.get("/download/{job_id}")
async def download(job_id: str) -> Any:
    """
    Download the transcription result.
    Returns 202 if still processing, 404 if job not found, 500 if failed.
    Returns the transcript file if completed.
    """

    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs_storage[job_id]
    status = job_data["status"]

    if status == JobStatus.PROCESSING:
        return JSONResponse(
            status_code=202,
            content={
                "message": "Transcription still in progress. Please try again in a few moments.",
                "status": "processing",
            },
        )

    elif status == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {job_data.get('error', 'Unknown error')}",
        )

    elif status == JobStatus.COMPLETED:
        artifacts_path = job_data.get("artifacts_path")

        if not artifacts_path or not os.path.exists(artifacts_path):
            raise HTTPException(status_code=404, detail="Transcript file not found")

        if os.path.isdir(artifacts_path):
            # If it's a directory, zip it first
            import shutil

            zip_path = f"{artifacts_path}.zip"
            if not os.path.exists(zip_path):
                shutil.make_archive(artifacts_path, "zip", artifacts_path)

            return FileResponse(
                path=zip_path,
                filename=f"transcript_{job_id}.zip",
                media_type="application/zip",
            )
        else:
            return FileResponse(
                path=artifacts_path,
                filename=f"transcript_{job_id}.txt",
                media_type="text/plain",
            )


async def process_transcription(
    job_id: str,
    file_data: Optional[Dict[str, Any]],
    youtube_url: Optional[str],
    language: str,
    engine: TranscriptionEngine,
    auto_correction: bool,
) -> None:
    """Background task to process transcription"""

    temp_audio_path: str

    try:
        logger.info(f"Starting transcription for job {job_id}")  # Debug log

        # Handle file upload
        if file_data:
            # Validate file type
            allowed_types = {
                "audio/mpeg",
                "audio/wav",
                "audio/mp4",
                "video/mp4",
                "video/quicktime",
                "audio/ogg",
                "video/webm",
            }
            content_type = file_data.get("content_type")
            if content_type and content_type not in allowed_types:
                raise Exception(f"Unsupported file type: {content_type}")

            # Save uploaded file temporarily
            filename = file_data.get("filename", "uploaded_file")
            temp_audio_path = f"data/{job_id}_{filename}"
            os.makedirs("data", exist_ok=True)

            with open(temp_audio_path, "wb") as buffer:
                buffer.write(file_data["content"])

            logger.info(f"Saved uploaded file: {temp_audio_path}")

        # Handle YouTube URL
        elif youtube_url:
            temp_audio_path = f"data/{job_id}_youtube_audio.wav"
            os.makedirs("data", exist_ok=True)

            logger.info(f"Downloading YouTube video: {youtube_url}")
            await download_youtube_video(youtube_url, temp_audio_path)
            logger.info(f"Downloaded to: {temp_audio_path}")

        # Initialize transcription service
        logger.info(f"Initializing transcription service with engine: {engine.value}")
        transcription_service = SpeachToText(
            preferred_engine=engine.value,
            auto_correction=auto_correction,
            language=language,
        )

        # Run transcription
        logger.info(f"Starting transcription of: {temp_audio_path}")
        transcription_result = await transcription_service.transcribe(
            video_path=Path(temp_audio_path),
            language=language,
            save_artifacts=True,
            return_only_text=False,
        )

        # Get artifacts directory path
        artifacts_dir = transcription_result.get("artifacts_dir", "")
        artifacts_path = f"data/{artifacts_dir}" if artifacts_dir else None

        logger.info(f"Transcription completed. Artifacts: {artifacts_path}")

        # Update job as completed
        jobs_storage[job_id].update(
            {
                "status": JobStatus.COMPLETED,
                "completed_at": datetime.now(),
                "result": transcription_result,
                "artifacts_path": artifacts_path,
            }
        )

        logger.info(f"Job {job_id} completed successfully")

        # Clean up temp file (keep artifacts)
        await asyncio.sleep(1)
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temp file: {temp_audio_path}")
            except Exception as e:
                logger.info(f"Failed to clean up {temp_audio_path}: {e}")

    except Exception as e:
        logger.info(f"Job {job_id} failed with error: {e}")

        # Update job as failed
        jobs_storage[job_id].update(
            {
                "status": JobStatus.FAILED,
                "completed_at": datetime.now(),
                "error": str(e),
            }
        )

        # Clean up on error
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
