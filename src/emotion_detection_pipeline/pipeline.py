import argparse
import os
import uuid
from pathlib import Path

import pandas as pd
import language_tool_python as ltp

from emotion_detection_pipeline.autocorrection.autocorrection import auto_correct_batch
from emotion_detection_pipeline.classifier.classify import EmotionPredictor
from emotion_detection_pipeline.logger import get_logger
from emotion_detection_pipeline.speach_to_text.stt_master import SpeachToText
from emotion_detection_pipeline.speach_to_text.utils.download_youtube_video import (
    download_youtube_video,
)
from emotion_detection_pipeline.translation.translate import translate_batch

logger = get_logger(__name__)
os.environ["PYTHONIOENCODING"] = "utf-8"


async def pipeline(
    file_path: Path,
    output_path: Path,
    autocorrection: bool = True,
    url_to_llm: str = "http://194.171.191.226:3061/api/generate",
) -> pd.DataFrame:
    # # Transcription
    transcription_service = SpeachToText(
        preferred_engine="assemblyai", auto_correction=False, language="pl"
    )

    logger.info("Starting transcription")
    result = await transcription_service.transcribe(
        video_path=file_path,
        language="pl",
        save_artifacts=True,
        return_only_text=False,
    )
    logger.info("Transcription completed")
    artifacts_dir = result.get("artifacts_dir", "")
    artifacts_path = f"{artifacts_dir}" if artifacts_dir else None
    logger.info(f"Artifacts saved at: {artifacts_path}")

    # Load the dataframe from the CSV file
    artifacts_path_obj = Path(artifacts_path)
    base_name = artifacts_path_obj.name.split("_")[0]
    csv_path = artifacts_path_obj / f"{base_name}_segments.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataframe with {len(df)} rows from {csv_path}")

    # Autocorrection (if enabled)
    if autocorrection:
        tool = ltp.LanguageTool("pl-PL")
        url = url_to_llm
        model = "qwen3:32b"

        logger.info("Starting autocorrection")
        df_corrected = auto_correct_batch(
            df,
            "Sentence",
            tool,
            url,
            model,
            max_workers=4,
        )
        df = df_corrected
        logger.info("Autocorrection completed")
        text_column_to_translate = "Sentence_corrected"
    else:
        text_column_to_translate = "Sentence"

    # Translation to English
    logger.info("Starting translation to English")
    translations = translate_batch(df[text_column_to_translate].tolist())
    df["Translation"] = translations
    logger.info("Translation completed")

    # Classification
    logger.info("Starting classification")
    predictor = EmotionPredictor()

    df = predictor.predict_dataframe(
        df,
        text_column="Translation",
        include_confidence=True,
    )
    logger.info("Classification completed")
    logger.info(
        f"Job finished. Dataframe has {len(df)} rows and columns: {df.columns.tolist()}"
    )

    # Save output
    output_path = output_path / f"{base_name}_processed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Processed dataframe saved to {output_path}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio/Video processing pipeline: transcription, autocorrection, translation, and emotion classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a local file with autocorrection
  python script.py --input video.mp4 --output ./results

  # Process YouTube video without autocorrection
  python script.py --youtube-url "https://www.youtube.com/watch?v=..." --output ./results --no-autocorrection

  # Process with custom LLM endpoint
  python script.py --input audio.wav --output ./results --llm-url "http://localhost:8080/api/generate"
        """,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", type=str, help="Path to input audio/video file"
    )
    input_group.add_argument(
        "-y", "--youtube-url", type=str, help="YouTube URL to download and process"
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory path for processed results",
    )

    # Autocorrection
    parser.add_argument(
        "--autocorrection",
        action="store_true",
        default=True,
        help="Enable autocorrection (default: enabled)",
    )
    parser.add_argument(
        "--no-autocorrection",
        dest="autocorrection",
        action="store_false",
        help="Disable autocorrection",
    )

    # LLM configuration
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://194.171.191.226:3061/api/generate",
        help="URL for LLM API endpoint (default: http://194.171.191.226:3061/api/generate)",
    )

    # Additional options
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded files (for YouTube) (default: ./data)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Setup output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine input file
    if args.youtube_url:
        # Download from YouTube
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_file = save_dir / f"{str(uuid.uuid4())}"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading YouTube video: {args.youtube_url}")
        title = await download_youtube_video(args.youtube_url, str(output_file))
        logger.info(f"Downloaded: {title}")

        if not output_file.suffix:
            output_file = output_file.with_suffix(".wav")

        input_file = output_file
    else:
        # Use provided input file
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Run pipeline
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Autocorrection: {args.autocorrection}")

    df = await pipeline(
        file_path=input_file,
        output_path=output_path,
        autocorrection=args.autocorrection,
        url_to_llm=args.llm_url,
    )

    logger.info("Pipeline completed successfully")
    return df


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
