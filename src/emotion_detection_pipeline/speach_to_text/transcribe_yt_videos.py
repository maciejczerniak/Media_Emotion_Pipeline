from pathlib import Path
from typing import Union


async def transcribe(video_dir: Union[Path, str]):

    # download videos from youtube (multithreaded)
    import asyncio
    from emotion_detection_pipeline.speach_to_text.stt_master import SpeachToText
    from tqdm import tqdm

    if isinstance(video_dir, str):
        video_dir = Path(video_dir)

    if video_dir.is_dir():
        video_paths = (
            list(video_dir.glob("*.mp3"))
            + list(video_dir.glob("*.mp4"))
            + list(video_dir.glob("*.wav"))
        )
    else:
        video_paths = [video_dir]
    if not video_paths:
        raise ValueError(f"No video files found in {video_dir}")

    stt_master = SpeachToText(preferred_engine="assemblyai", auto_correction=False)
    transcription_tasks = [
        stt_master.transcribe(video_path, save_artifacts=True)
        for video_path in video_paths
    ]
    transcriptions = []
    for f in tqdm(
        asyncio.as_completed(transcription_tasks), total=len(transcription_tasks)
    ):
        transcription = await f
        transcriptions.append(transcription)
    return transcriptions


if __name__ == "__main__":
    import asyncio

    video_dir = Path("../../../data/dataset_sadness_videos/videos/")
    results = asyncio.run(transcribe(video_dir))
    for result in results:
        print(result)
