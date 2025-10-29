from typing import Any
import pandas as pd
import spacy
import ast
from pandas import DataFrame
from spacy import Language

from task6.utils.text_preprocessing import process_all_features_batch


def process_polish_text_features(
    df: pd.DataFrame, text_column: str, nlp: Language, batch_size: int = 1000
) -> pd.DataFrame:
    """
    Process Polish text while preserving stop words, punctuation, and Polish characters.

    Args:
        df: DataFrame with text data
        text_column: Name of the text column to process
        nlp: spaCy language model
        batch_size: Batch size for processing

    Returns:
        DataFrame with processed text features
    """

    def process_text_batch(texts):
        """Process a batch of texts preserving Polish characters, stop words, and punctuation"""
        processed_texts = []

        for doc in nlp.pipe(texts, batch_size=batch_size):
            # Keep all tokens including stop words and punctuation
            # Only filter out spaces to maintain meaningful content
            tokens = [token.text for token in doc if not token.is_space]
            processed_texts.append(" ".join(tokens))

        return processed_texts

    # Process text in batches
    result_df = df.copy()
    texts = result_df[text_column].fillna("").astype(str).tolist()

    # Clean text using the twitter cleaning function if needed
    if any(
        text.startswith(("http", "www", "@", "#")) for text in texts[:100]
    ):  # Sample check
        texts = [twitter_clean_text(text) for text in texts]

    # Process in batches
    processed_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        processed_batch = process_text_batch(batch)
        processed_texts.extend(processed_batch)

    # Add processed text columns
    result_df["cleaned_text"] = texts  # Cleaned but not tokenized
    result_df["tokenized_text"] = processed_texts  # Tokenized preserving Polish chars

    # Add basic text statistics
    result_df["text_length"] = result_df["tokenized_text"].str.len()
    result_df["word_count"] = result_df["tokenized_text"].str.split().str.len()

    return result_df


def twitteremo_to_supported_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts Polish emotion dataset to GoEmotions-compatible format.

    Args:
        df: DataFrame with Polish emotion columns as binary indicators

    Returns:
        pd.DataFrame: DataFrame with 'text' and 'emotions' columns in GoEmotions format
    """

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Detect actual column names in the dataset (handle encoding variations)
    available_columns = result_df.columns.tolist()

    def find_polish_column(target_names):
        """Find Polish emotion column handling different encodings"""
        for col in available_columns:
            if col in target_names:
                return col
        return None

    # Polish emotion column variations to handle encoding issues
    emotion_mapping = {
        "radość": ["radość", "radoÅ›Ä‡", "radosc"],
        "smutek": ["smutek"],
        "zaufanie": ["zaufanie"],
        "wstręt": ["wstręt", "wstrÄ™t", "wstret"],
        "strach": ["strach"],
        "gniew": ["gniew"],
        "przeczuwanie": ["przeczuwanie"],
        "zdziwienie": ["zdziwienie"],
    }

    # Find actual column names in the dataset
    actual_emotion_columns = {}
    for emotion, variants in emotion_mapping.items():
        found_col = find_polish_column(variants)
        if found_col:
            actual_emotion_columns[emotion] = found_col

    # Polish to GoEmotions emotion mapping
    polish_to_goemotions = {
        "radość": ["joy", "amusement", "excitement", "optimism"],  # joy
        "smutek": ["sadness", "disappointment", "grief"],  # sadness
        "zaufanie": ["approval", "caring", "gratitude"],  # trust -> positive emotions
        "wstręt": ["disgust"],  # disgust
        "strach": ["fear", "nervousness"],  # fear
        "gniew": ["anger", "annoyance", "disapproval"],  # anger
        "przeczuwanie": [
            "curiosity",
            "realization",
        ],  # anticipation -> curiosity/awareness
        "zdziwienie": ["surprise"],  # surprise
    }

    # GoEmotions emotion to ID mapping (matching the order in prepare_data function)
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    emotion2id = {e: i for i, e in enumerate(emotions)}

    def convert_row_emotions(row):
        """Convert binary emotion indicators to GoEmotions ID list"""
        active_emotions = []

        # Check which Polish emotions are active (value = 1)
        for polish_emotion, column_name in actual_emotion_columns.items():
            if row.get(column_name, 0) == 1:
                # Map to GoEmotions and get the first (primary) mapping
                goemotions_list = polish_to_goemotions.get(polish_emotion, [])
                if goemotions_list:
                    # Use the first emotion as primary mapping
                    primary_emotion = goemotions_list[0]
                    if primary_emotion in emotion2id:
                        active_emotions.append(emotion2id[primary_emotion])

        # If no emotions found, default to neutral
        if not active_emotions:
            active_emotions = [emotion2id["neutral"]]

        return active_emotions

    # Apply conversion
    result_df["emotions"] = result_df.apply(convert_row_emotions, axis=1)

    # Rename text column if needed
    if "tekst" in result_df.columns:
        result_df["text"] = result_df["tekst"]

    # Keep only necessary columns for compatibility with prepare_data function
    final_columns = ["text", "emotions"]
    if "id" in result_df.columns:
        final_columns.insert(0, "id")
    if "data" in result_df.columns:
        final_columns.insert(-1, "data")

    # Select only existing columns
    available_columns = [col for col in final_columns if col in result_df.columns]
    result_df = result_df[available_columns]

    return result_df


def twitter_clean_text(text: str) -> str:
    """
    Cleans tweet text by removing URLs, mentions, hashtags, and extra whitespace.

    Args:
        text (str): The raw tweet text.
    Returns:
        str: The cleaned tweet text.
    """
    import re

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags (only the '#' symbol, keep the text)
    text = re.sub(r"#", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_data(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    nlp_model: str = "en_core_web_lg",
    return_artifacts: bool = False,
    dataset_type: str = "auto",
) -> (
    tuple[
        DataFrame,
        Language,
        dict[str, dict[str | Any, int] | dict[int, str | Any] | dict[str, int]],
    ]
    | tuple[DataFrame, list[str]]
):
    """
    Prepares the data for training, supporting both transcript and GoEmotions datasets.

    Args:
        df (pd.DataFrame): The input dataframe with raw data.
        text_column (str): The name of the column containing text data.
        label_column (str): The name of the column containing label data.
        nlp_model (str): The spaCy model to use for text processing. Default is "en_core_web_lg".
        return_artifacts (bool): Whether to return artifacts like encoders or scalers. Default is False.
        dataset_type (str): Type of dataset - "transcript", "goemotions", or "auto" for automatic detection.

    Returns:
        pd.DataFrame: The cleaned and prepared dataframe.
    """

    df = df.copy()
    df = df.dropna()

    # fixed mapping (do not change order!)
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    emotion2id = {e: i for i, e in enumerate(emotions)}
    id2emotion = {i: e for e, i in emotion2id.items()}

    # Auto-detect dataset type if not specified
    if dataset_type == "auto":
        sample_label = str(df[label_column].iloc[0])

        # Check if it's GoEmotions format (numeric IDs in brackets or direct numbers)
        if (
            sample_label.startswith("[") and sample_label.endswith("]")
        ) or sample_label.isdigit():
            dataset_type = "goemotions"
        # Check if it's transcript format (direct emotion names)
        elif sample_label in [
            "anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "neutral",
            "happiness",
        ]:
            dataset_type = "transcript"
        else:
            # Default to GoEmotions for numeric data
            if df[label_column].dtype in [int, float]:
                dataset_type = "goemotions"
            else:
                dataset_type = "transcript"

    print(f"Detected dataset type: {dataset_type}")

    # Handle different dataset formats
    if dataset_type == "transcript":
        # Transcript dataset: direct Ekman emotion names
        transcript_to_ekman = {
            "happiness": "joy",
            "anger": "anger",
            "sadness": "sadness",
            "fear": "fear",
            "disgust": "disgust",
            "surprise": "surprise",
            "neutral": "neutral",
        }

        # Map transcript emotions directly to Ekman categories
        df["ekman_emotion"] = df[label_column].map(transcript_to_ekman)
        # Fill any unmapped values with neutral
        df["ekman_emotion"] = df["ekman_emotion"].fillna("neutral")

    elif dataset_type == "goemotions":
        # GoEmotions dataset: numeric IDs that need to be mapped to fine emotions then Ekman

        # Parse numeric labels from various formats
        def parse_emotion_ids(label):
            if isinstance(label, str):
                if label.startswith("[") and label.endswith("]"):
                    # Format like "[27]" or "[2, 5]"
                    try:
                        return ast.literal_eval(label)
                    except:
                        return [int(label.strip("[]"))]
                else:
                    # Single number as string
                    return [int(label)]
            elif isinstance(label, (int, float)):
                # Direct numeric
                return [int(label)]
            else:
                # List or other format
                return label if isinstance(label, list) else [label]

        df[label_column] = df[label_column].apply(parse_emotion_ids)

        # Convert IDs to emotion names
        df["emotion_text"] = df[label_column].apply(
            lambda x: [
                id2emotion.get(i, "neutral")
                for i in (x if isinstance(x, list) else [x])
            ]
        )

        # Map emotions to Ekman's categories
        emotion_ekman_mapping = {
            "anger": ["anger", "annoyance", "disapproval"],
            "disgust": ["disgust"],
            "fear": ["fear", "nervousness"],
            "joy": [
                "joy",
                "amusement",
                "approval",
                "excitement",
                "gratitude",
                "love",
                "optimism",
                "relief",
                "pride",
                "admiration",
                "desire",
                "caring",
            ],
            "sadness": [
                "sadness",
                "disappointment",
                "embarrassment",
                "grief",
                "remorse",
            ],
            "surprise": ["surprise", "realization", "confusion", "curiosity"],
            "neutral": ["neutral"],
        }

        df["ekman_emotions"] = df["emotion_text"].apply(
            lambda x: [
                k for k, v in emotion_ekman_mapping.items() if any(i in v for i in x)
            ]
        )

        # Leave only one ekman emotion per row (the first one, fallback = neutral)
        df["ekman_emotion"] = df["ekman_emotions"].apply(
            lambda x: x[0] if len(x) > 0 else "neutral"
        )

    # process text column
    nlp = spacy.load(nlp_model, disable=["ner", "parser"])
    df = process_all_features_batch(df, text_column, nlp, batch_size=5000)

    # Create final Ekman mapping → integer ids (stable order)
    ekman_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
    ekman2id = {e: i for i, e in enumerate(ekman_labels)}

    # Convert final emotions to numeric IDs
    df["ekman_emotion"] = df["ekman_emotion"].map(ekman2id)

    df = df.reset_index(drop=True)

    if return_artifacts:
        artifacts = {
            "fine_emotion2id": emotion2id,
            "fine_id2emotion": id2emotion,
            "ekman_emotion2id": ekman2id,
        }

        if dataset_type == "goemotions":
            artifacts["ekman_mapping"] = emotion_ekman_mapping
        elif dataset_type == "transcript":
            artifacts["transcript_to_ekman"] = transcript_to_ekman

        return df, nlp, artifacts

    # drop intermediate columns
    columns_to_drop = ["cleaned_text"]
    if "emotion_text" in df.columns:
        columns_to_drop.append("emotion_text")
    if "ekman_emotions" in df.columns:
        columns_to_drop.append("ekman_emotions")
    if label_column in df.columns and label_column != "ekman emotion":
        columns_to_drop.append(label_column)

    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df = df.drop(columns=existing_columns_to_drop)

    return df, ekman_labels
