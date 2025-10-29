from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, List, Dict, Tuple
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_MODEL_PATH = SCRIPT_DIR / "roberta_emotion_model"

class EmotionPredictor:
    """
    RoBERTa-based emotion classifier for 7 emotions:
    anger, disgust, fear, joy, neutral, sadness, surprise
    """

    def __init__(self, model_path: str = str(DEFAULT_MODEL_PATH)):
        """
        Initialize the emotion predictor.

        Args:
            model_path: Path to the saved model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model from the reliable path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )

        # *** YOUR FIX IS ADDED HERE ***
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            attn_implementation="eager",  # Resolves SDPA warnings
        )

        self.model.to(self.device)
        self.model.eval()

        # Emotion labels (must match training order)
        self.emotion_names = [
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise",
        ]

        print(f"✓ Model loaded from {model_path}")

    def predict(self, text: str) -> str:
        """
        Predict emotion for a single text.

        Args:
            text: Input text

        Returns:
            Predicted emotion label
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        return self.emotion_names[prediction]

    def predict_with_confidence(self, text: str) -> dict:
        """
        Predict emotion with confidence scores for all emotions.

        Args:
            text: Input text

        Returns:
            Dictionary with predicted emotion and all confidence scores
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            prediction = torch.argmax(logits, dim=1).item()

        # Create confidence dict
        confidences = {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_names, probabilities.cpu())
        }

        return {
            "predicted_emotion": self.emotion_names[prediction],
            "confidence": float(probabilities[prediction].cpu()),
            "all_confidences": confidences,
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 16, return_confidence: bool = False
    ) -> Union[List[str], Tuple[List[str], List[float]]]:
        """
        Predict emotions for multiple texts efficiently.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            return_confidence: If True, return confidence scores as well

        Returns:
            If return_confidence=False: List of predicted emotion labels
            If return_confidence=True: Tuple of (predictions, confidences)
        """
        predictions = []
        confidences = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize batch
            encoding = self.tokenizer(
                batch,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                batch_confidences = probabilities.max(dim=1).values.cpu().numpy()

            # Convert to emotion names
            batch_emotions = [self.emotion_names[pred] for pred in batch_predictions]
            predictions.extend(batch_emotions)
            confidences.extend(batch_confidences.tolist())

        if return_confidence:
            return predictions, confidences
        return predictions

    def predict_batch_with_all_confidences(
        self, texts: List[str], batch_size: int = 16
    ) -> List[Dict[str, Union[str, float, Dict]]]:
        """
        Predict emotions for multiple texts with full confidence distributions.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of dictionaries, each containing:
            - predicted_emotion: str
            - confidence: float (confidence of predicted emotion)
            - all_confidences: dict (confidence for each emotion)
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize batch
            encoding = self.tokenizer(
                batch,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

            # Process each item in batch
            for idx, (pred_idx, probs) in enumerate(
                zip(batch_predictions, probabilities)
            ):
                confidences = {
                    emotion: float(prob)
                    for emotion, prob in zip(self.emotion_names, probs.cpu())
                }

                results.append(
                    {
                        "predicted_emotion": self.emotion_names[pred_idx],
                        "confidence": float(probs[pred_idx].cpu()),
                        "all_confidences": confidences,
                    }
                )

        return results

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "Emotion_core",
        include_confidence: bool = True,
        include_all_confidences: bool = False,
    ) -> pd.DataFrame:
        """
        Predict emotions for a DataFrame and add prediction columns.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            include_confidence: If True, add confidence column
            include_all_confidences: If True, add columns for all emotion confidences

        Returns:
            DataFrame with added prediction columns
        """
        texts = df[text_column].tolist()

        if include_all_confidences:
            # Get full confidence distributions
            results = self.predict_batch_with_all_confidences(texts)

            # Add predicted emotion and main confidence
            df[label_column] = [r["predicted_emotion"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]

            # Add confidence columns for each emotion
            for emotion in self.emotion_names:
                df[f"conf_{emotion}"] = [r["all_confidences"][emotion] for r in results]

        elif include_confidence:
            # Get predictions with confidence
            predictions, confidences = self.predict_batch(texts, return_confidence=True)
            df[label_column] = predictions
            df["confidence"] = confidences

        else:
            # Get predictions only
            predictions = self.predict_batch(texts, return_confidence=False)
            df[label_column] = predictions

        return df
