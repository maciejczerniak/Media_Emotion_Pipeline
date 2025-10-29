from typing import Optional, Tuple, Union

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(
    df: pd.DataFrame,
    column: str,
    nlp_model: str = "en_core_web_lg",
    sparse: bool = True,
    max_features: Optional[int] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    return_vectorizer: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TfidfVectorizer]]:
    """
    Create TF-IDF matrix for text data with option to reuse vectorizer.

    Args:
        df: DataFrame containing text data
        column: Name of column containing text data
        nlp_model: spaCy model to use for text processing
        sparse: Whether to return sparse format (not implemented)
        max_features: Maximum number of features for TF-IDF
        vectorizer: Pre-fitted TfidfVectorizer to reuse (optional)
        return_vectorizer: Whether to return the vectorizer along with DataFrame

    Returns:
        DataFrame with TF-IDF features added, optionally with vectorizer
    """
    nlp = spacy.load(nlp_model, disable=["ner", "parser"])

    # Process text data based on input format
    if df[column].apply(lambda x: isinstance(x, str)).all():
        # String column
        docs = list(nlp.pipe(df[column].astype("string").tolist()))
    elif (
        df[column]
        .apply(
            lambda x: isinstance(x, list) and all(isinstance(item, str) for item in x)
        )
        .all()
    ):
        # List of strings column
        docs = list(nlp.pipe([" ".join(doc) for doc in df[column]]))
    elif (
        df[column]
        .apply(
            lambda x: isinstance(x, list)
            and all(
                isinstance(item, list)
                and all(isinstance(subitem, str) for subitem in item)
                for item in x
            )
        )
        .all()
    ):
        # List of lists of strings column
        docs = list(
            nlp.pipe(
                [
                    " ".join([subitem for item in doc for subitem in item])
                    for doc in df[column]
                ]
            )
        )
    else:
        raise ValueError(
            "Column must be of type string, list of strings, or list of lists of strings."
        )

    # Extract text content
    text_content = [doc.text for doc in docs]

    # Handle vectorizer creation or reuse
    if vectorizer is None:
        # Create new vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 1),  # Can be modified as needed
            min_df=1,
            max_df=1.0,
        )
        # Fit and transform
        X = vectorizer.fit_transform(text_content)
    else:
        # Use existing vectorizer (already fitted)
        X = vectorizer.transform(text_content)

    if sparse:
        # Store each row as a sparse vector in a list
        raise NotImplementedError("Sparse format not implemented yet.")
    else:
        # Convert to dense list of lists
        tfidf_matrix = X.toarray().tolist()

    # Add TF-IDF column to dataframe
    df_result = df.copy()
    df_result[f"{column}_tfidf"] = tfidf_matrix

    if return_vectorizer:
        return df_result, vectorizer
    else:
        return df_result


def create_unified_tfidf_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
    nlp_model: str = "en_core_web_lg",
    max_features: Optional[int] = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """
    Create TF-IDF matrices for training and test data using the same vectorizer.

    This function solves the feature mismatch problem by ensuring both datasets
    use the same vocabulary and feature space.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        column: Name of text column
        nlp_model: spaCy model name
        max_features: Maximum number of TF-IDF features

    Returns:
        Tuple of (processed_train_df, processed_test_df, fitted_vectorizer)
    """

    # First pass: create vectorizer using training data
    print("Fitting TF-IDF vectorizer on training data...")
    train_processed, vectorizer = create_tfidf_matrix(
        train_df,
        column=column,
        nlp_model=nlp_model,
        sparse=False,
        max_features=max_features,
        return_vectorizer=True,
    )

    # Second pass: transform test data using the same vectorizer
    print("Transforming test data with same vectorizer...")
    test_processed = create_tfidf_matrix(
        test_df,
        column=column,
        nlp_model=nlp_model,
        sparse=False,
        vectorizer=vectorizer,  # Reuse the fitted vectorizer
    )

    # Verify feature dimensions match
    train_features = len(train_processed[f"{column}_tfidf"].iloc[0])
    test_features = len(test_processed[f"{column}_tfidf"].iloc[0])

    print(f"Training features: {train_features}")
    print(f"Test features: {test_features}")
    print(f"✓ Feature dimensions match: {train_features == test_features}")

    if train_features != test_features:
        raise ValueError(
            f"Feature mismatch: train={train_features}, test={test_features}"
        )

    return train_processed, test_processed, vectorizer
