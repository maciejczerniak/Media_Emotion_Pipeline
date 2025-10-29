from transformers import MarianMTModel, MarianTokenizer
import torch


def translate_pl_to_en(
    text: str, model_name: str = "Helsinki-NLP/opus-mt-pl-en"
) -> str:
    """
    Translate Polish text to English using HuggingFace model.

    Args:
        text: Polish text to translate
        model_name: HuggingFace model name

    Returns:
        Translated English text
    """
    # Load model and tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Tokenize
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated


def translate_batch(
    texts: list[str],
    model_name: str = "Helsinki-NLP/opus-mt-pl-en",
    batch_size: int = 32,
) -> list[str]:
    """
    Translate multiple Polish texts to English in batches.

    Args:
        texts: List of Polish texts
        model_name: HuggingFace model name
        batch_size: Number of texts to process at once

    Returns:
        List of translated English texts
    """
    # Load model and tokenizer once
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    translations = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate translations
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512)

        # Decode all outputs in batch
        batch_translations = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        translations.extend(batch_translations)

    return translations


if __name__ == "__main__":
    # Example usage
    polish_text = "Dzień dobry! Jak się masz?"
    english_text = translate_pl_to_en(polish_text)
    print(f"PL: {polish_text}")
    print(f"EN: {english_text}")

    # Batch example
    polish_texts = [
        "Witaj świecie!",
        "Python jest świetnym językiem programowania.",
        "Studiuję uczenie maszynowe na uniwersytecie.",
    ]

    english_texts = translate_batch(polish_texts)

    print("\nBatch translation:")
    for pl, en in zip(polish_texts, english_texts):
        print(f"PL: {pl}")
        print(f"EN: {en}")
        print()