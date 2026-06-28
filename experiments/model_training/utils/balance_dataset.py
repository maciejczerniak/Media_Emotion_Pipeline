import random
from typing import List

import nltk
import pandas as pd
from nltk.corpus import wordnet
import tqdm

nltk.download("wordnet")


class EDAugmenter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Percentage of words to change

    def synonym_replacement(self, words, n):
        if len(words) == 0:
            return words

        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [
                    synonym if word == random_word else word for word in new_words
                ]
                num_replaced += 1
            if num_replaced >= n:
                break
        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char.isalpha()])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def random_deletion(self, words, p=0.1):
        if len(words) <= 1:  # Fixed: handle empty lists
            return words
        new_words = [word for word in words if random.uniform(0, 1) > p]
        return (
            new_words if len(new_words) > 0 else words
        )  # Fixed: return original if all deleted

    def random_swap(self, words, n=1):
        if len(words) <= 1:  # Fixed: can't swap with fewer than 2 words
            return words

        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        if len(new_words) <= 1:
            return new_words

        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = (
            new_words[random_idx_2],
            new_words[random_idx_1],
        )
        return new_words

    def random_insertion(self, words, n=1):
        if len(words) == 0:  # Fixed: can't insert into empty list
            return words

        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        if len(new_words) == 0:
            return

        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words))  # Allow insertion at end
        new_words.insert(random_idx, random_synonym)


# Apply to minority classes only
def augment_minority_classes(df, classes_to_augment: List[int], target_samples=2000):
    augmenter = EDAugmenter()
    augmented_data = []

    # Focus on fear (2) and disgust (1)
    for emotion_id in classes_to_augment:
        emotion_samples = df[df["ekman_emotion"] == emotion_id]
        current_count = len(emotion_samples)
        needed = target_samples - current_count

        print(f"Augmenting emotion {emotion_id}: {current_count} → {target_samples}")

        for i in range(needed):
            # Sample from existing data
            sample = emotion_samples.sample(1).iloc[0]
            text_tokens = sample["lemmatized_text"]

            # Skip if tokens are empty or too short
            if len(text_tokens) <= 1:
                continue

            # Apply random EDA technique
            technique = random.choice(["synonym", "insertion", "swap", "deletion"])
            n = max(1, int(len(text_tokens) * 0.1))  # 10% of words

            try:
                if technique == "synonym":
                    new_tokens = augmenter.synonym_replacement(text_tokens, n)
                elif technique == "insertion":
                    new_tokens = augmenter.random_insertion(text_tokens, n)
                elif technique == "swap":
                    new_tokens = augmenter.random_swap(text_tokens, n)
                else:  # deletion
                    new_tokens = augmenter.random_deletion(text_tokens)

                # Skip if augmentation failed or resulted in empty tokens
                if len(new_tokens) == 0:
                    continue

                # Create new sample
                new_sample = sample.copy()
                new_sample["lemmatized_text"] = new_tokens
                new_sample["text"] = " ".join(new_tokens)  # Reconstructed text
                augmented_data.append(new_sample)

            except Exception as e:
                print(f"Skipping sample due to error: {e}")
                continue

    # print a few augmented samples for verification
    for i in range(min(3, len(augmented_data))):
        print(f"Augmented Sample {i + 1}: {augmented_data[i]['text']}")

    return pd.DataFrame(augmented_data)


def undersample_majority_classes(df, classes_to_reduce: List[int], target_samples=2000):
    reduced_data = []

    for emotion_id in classes_to_reduce:
        emotion_samples = df[df["ekman_emotion"] == emotion_id]
        current_count = len(emotion_samples)

        print(f"Reducing emotion {emotion_id}: {current_count} → {target_samples}")

        if current_count > target_samples:
            reduced_samples = emotion_samples.sample(target_samples, random_state=42)
        else:
            reduced_samples = emotion_samples

        reduced_data.append(reduced_samples)

    # Combine with non-reduced classes
    non_reduced_classes = df[~df["ekman_emotion"].isin(classes_to_reduce)]
    reduced_data.append(non_reduced_classes)

    return pd.concat(reduced_data, ignore_index=True)


def augment_minority_class_back_translation(
    df,
    classes_to_augment: List[int],
    target_samples=2000,
    text_column="Translation",
    batch_size=32,
):
    import os

    os.environ.setdefault("USE_TF", "0")

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load translation models
    model_name_en_fr = "Helsinki-NLP/opus-mt-en-de"
    model_name_fr_en = "Helsinki-NLP/opus-mt-de-en"

    tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
    model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr).to(device)

    tokenizer_fr_en = MarianTokenizer.from_pretrained(model_name_fr_en)
    model_fr_en = MarianMTModel.from_pretrained(model_name_fr_en).to(device)

    # Set models to evaluation mode for faster inference
    model_en_fr.eval()
    model_fr_en.eval()

    def translate_batch(texts, tokenizer, model):
        """Translate a batch of texts"""
        with torch.no_grad():  # Disable gradient computation for inference
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            translated = model.generate(**inputs, max_length=512)
            return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    augmented_data = []

    for emotion_id in classes_to_augment:
        emotion_samples = df[df["ekman_emotion"] == emotion_id]
        current_count = len(emotion_samples)
        needed = target_samples - current_count
        cap = 2 * current_count  # Avoid excessive augmentation
        if needed > cap:
            needed = cap

        print(
            f"Back-Translating emotion {emotion_id}: {current_count} → {target_samples} (batch_size={batch_size})"
        )

        # Sample all needed texts at once
        sampled_indices = [emotion_samples.sample(1).index[0] for _ in range(needed)]

        # Process in batches
        for i in tqdm.tqdm(
            range(0, needed, batch_size), desc=f"Augmenting emotion {emotion_id}"
        ):
            batch_end = min(i + batch_size, needed)
            batch_indices = sampled_indices[i:batch_end]
            batch_samples = emotion_samples.loc[batch_indices]

            try:
                # Get batch of texts
                texts_batch = batch_samples[text_column].tolist()

                # English to German (batch)
                de_texts = translate_batch(texts_batch, tokenizer_en_fr, model_en_fr)

                # German back to English (batch)
                back_translated_texts = translate_batch(
                    de_texts, tokenizer_fr_en, model_fr_en
                )

                # Create new samples
                for idx, (_, sample) in enumerate(batch_samples.iterrows()):
                    new_sample = sample.copy()
                    new_sample["text"] = back_translated_texts[idx]
                    new_sample["lemmatized_text"] = back_translated_texts[
                        idx
                    ].split()  # Simple tokenization
                    augmented_data.append(new_sample)

            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                continue

    # print a few augmented samples for verification
    for i in range(min(3, len(augmented_data))):
        print(f"Back-Translated Sample {i + 1}: {augmented_data[i]['text']}")

    return pd.DataFrame(augmented_data)
