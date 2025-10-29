from typing import Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from numpy import ndarray
from pandas import Series
from tqdm import tqdm


def prepare_rnn_data(
    df: pd.DataFrame,
    lemma_column: str,
    embedding_model,
    emotion_column: str,
    max_length: int = 50,
    verbose: bool = True,
) -> Tuple[ndarray, Series]:
    if verbose:
        print(f"Preparing RNN data from {len(df)} samples...")
        print(f"Embedding dimension: {embedding_model.vector_size}")
        print(f"Max sequence length: {max_length}")

    y = df[emotion_column]

    X_sequences = []
    oov_count = 0
    total_tokens = 0

    for i, token_list in enumerate(
        tqdm(df[lemma_column], desc="Converting to embeddings")
    ):
        if isinstance(token_list, str):
            # Always use split() - it's safer and matches our data format
            token_list = token_list.split()
        elif not isinstance(token_list, list):
            # Fallback for unexpected types
            token_list = []

        # Convert tokens to embeddings
        sequence_embeddings = []

        for token in token_list[:max_length]:
            total_tokens += 1
            if token in embedding_model:
                sequence_embeddings.append(embedding_model[token])
            else:
                # Handle out-of-vocabulary words
                sequence_embeddings.append(np.zeros(embedding_model.vector_size))
                oov_count += 1

        # Pad short sequences with zeros
        while len(sequence_embeddings) < max_length:
            sequence_embeddings.append(np.zeros(embedding_model.vector_size))

        X_sequences.append(np.array(sequence_embeddings))

    X = np.array(X_sequences)

    if verbose:
        print(f"✓ Final shape: {X.shape}")
        print(
            f"✓ OOV rate: {oov_count / total_tokens:.2%} ({oov_count}/{total_tokens})"
        )
        print(f"✓ Data type: {X.dtype}")

    return X, y


def load_embeddings(model_path):
    """Load your word embedding model"""
    try:
        # Try loading as KeyedVectors (if saved that way)
        model = KeyedVectors.load(model_path)
        print(f"Loaded KeyedVectors: {model.vector_size}D, {len(model)} words")
        return model
    except:
        try:
            # Try loading as Word2Vec model
            from gensim.models import Word2Vec

            model = Word2Vec.load(model_path)
            print(f"Loaded Word2Vec: {model.wv.vector_size}D, {len(model.wv)} words")
            return model.wv
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


if __name__ == "__main__":
    import gensim.downloader as api

    embedding_model = api.load("glove-twitter-100")

    # Example usage
    data = {
        "lemma": [["happy", "day"], ["sad", "night"], ["joyful", "moment"]],
        "emotion": ["joy", "sadness", "joy"],
    }
    df = pd.DataFrame(data)
    X, y = prepare_rnn_data(df, "lemma", embedding_model, "emotion", max_length=5)
    print(X)
