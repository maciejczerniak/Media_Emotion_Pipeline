from pathlib import Path
from typing import Dict, Any

from emotion_detection_pipeline.logger import get_logger

logger = get_logger(__name__)


def _save_artifacts(video_path: Path, transcription_result: Dict[str, Any]) -> Path:
    """
    Save transcription artifacts to files.
    Args:
        video_path: Path to the original video/audio file.
        transcription_result: The result dictionary from the transcription process.

    Note: The transcription_result is expected to have the following structure:
        {
            "full_text": str,
            "segments_df": pd.DataFrame,
        }
    Returns: Path to the directory where artifacts are saved.
    """
    artifacts_dir = video_path.parent / f"{video_path.stem}_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save full text
    full_text_path = artifacts_dir / f"{video_path.stem}_transcription.txt"
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(transcription_result["full_text"])

    logger.info(f"Transcription text saved to {full_text_path}")

    # Save segments DataFrame if available
    if "segments_df" in transcription_result:
        segments_df_path = artifacts_dir / f"{video_path.stem}_segments.csv"
        transcription_result["segments_df"].to_csv(segments_df_path, index=False)
        logger.info(f"Transcription segments saved to {segments_df_path}")
    return artifacts_dir
