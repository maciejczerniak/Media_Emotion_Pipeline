import pandas as pd
import torch

from media_emotion_pipeline.classifier import classify


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        length = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.zeros((length, 2), dtype=torch.long),
            "attention_mask": torch.ones((length, 2), dtype=torch.long),
        }


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return None

    def __call__(self, input_ids, attention_mask):
        rows = input_ids.shape[0]
        logits = torch.zeros((rows, 7))
        logits[:, 3] = 10
        return type("Output", (), {"logits": logits})


def test_emotion_predictor_predicts_dataframe(monkeypatch) -> None:
    monkeypatch.setattr(
        classify.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        classify.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(classify.torch.cuda, "is_available", lambda: False)

    predictor = classify.EmotionPredictor(model_path="unused")
    df = predictor.predict_dataframe(
        pd.DataFrame({"Translation": ["hello", "world"]}),
        text_column="Translation",
    )

    assert df["Emotion_core"].tolist() == ["joy", "joy"]
    assert df["confidence"].tolist() == [df["confidence"].iloc[0]] * 2


def test_emotion_predictor_single_prediction_methods(monkeypatch) -> None:
    monkeypatch.setattr(
        classify.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        classify.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(classify.torch.cuda, "is_available", lambda: False)

    predictor = classify.EmotionPredictor(model_path="unused")

    assert predictor.predict("hello") == "joy"
    result = predictor.predict_with_confidence("hello")
    assert result["predicted_emotion"] == "joy"
    assert set(result["all_confidences"]) == set(predictor.emotion_names)


def test_emotion_predictor_dataframe_variants(monkeypatch) -> None:
    monkeypatch.setattr(
        classify.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        classify.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(classify.torch.cuda, "is_available", lambda: False)

    predictor = classify.EmotionPredictor(model_path="unused")

    with_all = predictor.predict_dataframe(
        pd.DataFrame({"text": ["hello"]}),
        include_all_confidences=True,
    )
    assert with_all["Emotion_core"].tolist() == ["joy"]
    assert "conf_joy" in with_all.columns

    without_confidence = predictor.predict_dataframe(
        pd.DataFrame({"text": ["hello"]}),
        include_confidence=False,
    )
    assert without_confidence["Emotion_core"].tolist() == ["joy"]
    assert "confidence" not in without_confidence.columns
