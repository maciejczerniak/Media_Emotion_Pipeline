import re
from typing import List

import pandas as pd
from spacy import Language


def tokenize(text: str, nlp) -> List[str]:
    """Tokenizes a given text into a list of words.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        List[str]: A list of tokenized words.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    doc = nlp(text)
    return [token.text for token in doc]


def lemmatize(text: str, nlp) -> List[str]:
    """Lemmatizes a given text into a list of lemmas.

    Args:
        text (str): The input text to be lemmatized.

    Returns:
        List[str]: A list of lemmatized words.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    doc = nlp(text)
    return [token.lemma_ for token in doc]


def preprocess_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove user mentions and subreddits
    text = re.sub(r"@\w+|r/\w+|u/\w+", "", text)

    # Remove special characters but keep spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text_vectorized(texts: List[str]) -> List[str]:
    """Vectorized preprocessing - much faster than individual apply()"""
    cleaned_texts = []
    for text in texts:
        # Convert to lowercase
        text = text.lower()
        # Remove URLs, mentions, special chars
        text = re.sub(r"http\S+|www\S+|https\S+|@\w+|r/\w+|u/\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned_texts.append(text)
    return cleaned_texts


def process_all_features_batch(
    df: pd.DataFrame, text_column: str, nlp: Language, batch_size: int = 1000
) -> pd.DataFrame:
    """Process all NLP features in one optimized pass"""

    print("Starting batch preprocessing...")

    # Step 1: Vectorized text cleaning (very fast)
    df["cleaned_text"] = preprocess_text_vectorized(df[text_column].tolist())
    print("✓ Text cleaning completed")

    # Step 2: Batch spaCy processing (much faster than individual apply)
    all_tokens = []
    all_lemmas = []

    # Process in batches to avoid memory issues
    texts = df["cleaned_text"].tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Process batch through spaCy pipeline
        docs = list(nlp.pipe(batch_texts, batch_size=batch_size))

        # Extract tokens and lemmas from batch
        batch_tokens = []
        batch_lemmas = []

        for doc in docs:
            tokens = [token.text for token in doc]
            lemmas = [token.lemma_ for token in doc]
            batch_tokens.append(tokens)
            batch_lemmas.append(lemmas)

        all_tokens.extend(batch_tokens)
        all_lemmas.extend(batch_lemmas)

        if (i // batch_size + 1) % 10 == 0:
            print(f"✓ Processed {i + len(batch_texts)} / {len(texts)} documents")

    df["tokenized_text"] = all_tokens
    df["lemmatized_text"] = all_lemmas

    print("✓ All NLP processing completed")
    return df
