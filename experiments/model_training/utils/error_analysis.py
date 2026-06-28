import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import random


def analyze_classification_errors(
    y_true,
    y_pred,
    class_names=None,
    sample_texts=None,
    model=None,
    X_test=None,
    top_k_errors=5,
    figsize=(12, 8),
):
    """
    Universal error analysis function for classification models.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes. If None, uses numeric labels
    sample_texts : array-like, optional
        Original text samples for qualitative analysis
    model : object, optional
        Trained model (for confidence analysis if supports decision_function/predict_proba)
    X_test : array-like, optional
        Test features (required if model provided)
    top_k_errors : int, default=5
        Number of top error patterns to show
    figsize : tuple, default=(12, 8)
        Figure size for plots

    Returns:
    --------
    dict : Error analysis results and metrics
    """

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Set up class names
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_labels]

    n_classes = len(unique_labels)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # 2. Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
        digits=3,
    )

    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0, digits=3
        )
    )

    # 3. Error Pattern Analysis
    misclassified_mask = y_true != y_pred
    error_patterns = defaultdict(int)

    for i in range(len(y_true)):
        if misclassified_mask[i]:
            true_class = class_names[unique_labels.index(y_true[i])]
            pred_class = class_names[unique_labels.index(y_pred[i])]
            error_patterns[f"{true_class} → {pred_class}"] += 1

    # Sort error patterns by frequency
    top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[
        :top_k_errors
    ]

    print(f"\nTop {top_k_errors} Error Patterns: (true → predicted)")
    print("-" * 30)
    for pattern, count in top_errors:
        total_errors = sum(error_patterns.values())
        print(f"{pattern}: {count} ({count / total_errors:.1%} of errors)")

    # 4. Per-class Error Analysis
    print("\nPer-Class Performance:")
    print("-" * 30)

    class_metrics = []
    for i, class_name in enumerate(class_names):
        label = unique_labels[i]

        # Class-specific metrics
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        true_negatives = cm.sum() - true_positives - false_positives - false_negatives

        # Calculate metrics
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        support = (y_true == label).sum()

        class_metrics.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "errors": support - true_positives,
            }
        )

        print(
            f"{class_name:12} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}, "
            f"Support: {support:3d}, Errors: {support - true_positives:3d}"
        )

    # 5. Confidence Analysis (if model provided)
    confidence_analysis = None
    if model is not None and X_test is not None:
        confidence_analysis = _analyze_confidence(
            model, X_test, y_true, y_pred, class_names
        )

    # 6. Sample Error Examples
    if sample_texts is not None:
        print("\nSample Error Examples:")
        print("-" * 30)
        _show_error_examples(
            y_true, y_pred, sample_texts, class_names, unique_labels, top_errors[:3]
        )

    # 7. Overall Statistics
    accuracy = (y_true == y_pred).mean()
    total_errors = misclassified_mask.sum()

    print("\nOverall Statistics:")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.3f}")
    print(
        f"Total errors: {total_errors}/{len(y_true)} ({total_errors / len(y_true):.1%})"
    )
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")

    # Return results
    results = {
        "confusion_matrix": cm,
        "classification_report": report,
        "error_patterns": dict(error_patterns),
        "top_errors": top_errors,
        "class_metrics": class_metrics,
        "misclassified_indices": np.where(misclassified_mask)[0],
        "accuracy": accuracy,
        "confidence_analysis": confidence_analysis,
    }

    return results


def _analyze_confidence(model, X_test, y_true, y_pred, class_names):
    """Helper function to analyze prediction confidence"""
    try:
        # Try decision_function first (SVM, LogisticRegression)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            if scores.ndim == 1:  # Binary classification
                confidence = np.abs(scores)
            else:  # Multi-class - use margin between top two scores
                confidence = (
                    np.max(scores, axis=1) - np.partition(scores, -2, axis=1)[:, -2]
                )

        # Try predict_proba (most classifiers)
        elif hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            confidence = np.max(probas, axis=1)

        else:
            print("Model doesn't support confidence analysis")
            return None

        # Analyze confidence for correct vs incorrect predictions
        correct_mask = y_true == y_pred
        correct_confidence = confidence[correct_mask]
        incorrect_confidence = confidence[~correct_mask]

        print("\nConfidence Analysis:")
        print("-" * 30)
        print(f"Average confidence (correct): {correct_confidence.mean():.3f}")
        print(f"Average confidence (incorrect): {incorrect_confidence.mean():.3f}")

        # Find concerning cases
        high_conf_errors = np.where(
            ~correct_mask & (confidence > np.percentile(confidence, 80))
        )[0]
        low_conf_correct = np.where(
            correct_mask & (confidence < np.percentile(confidence, 20))
        )[0]

        print(f"High-confidence errors: {len(high_conf_errors)}")
        print(f"Low-confidence correct: {len(low_conf_correct)}")

        return {
            "confidence_scores": confidence,
            "correct_confidence_mean": correct_confidence.mean(),
            "incorrect_confidence_mean": incorrect_confidence.mean(),
            "high_confidence_errors": high_conf_errors,
            "low_confidence_correct": low_conf_correct,
        }

    except Exception as e:
        print(f"Confidence analysis failed: {e}")
        return None


def _show_error_examples(
    y_true, y_pred, sample_texts, class_names, unique_labels, top_error_patterns
):
    """Helper function to show sample error examples"""

    for pattern, count in top_error_patterns:
        if count < 2:  # Skip if too few examples
            continue

        true_class, pred_class = pattern.split(" → ")
        true_idx = class_names.index(true_class)
        pred_idx = class_names.index(pred_class)
        true_label = unique_labels[true_idx]
        pred_label = unique_labels[pred_idx]

        # Find examples of this error pattern
        error_indices = np.where((y_true == true_label) & (y_pred == pred_label))[0]

        print(f"\n{pattern} ({count} cases):")
        sample_count = min(2, len(error_indices))
        sample_indices = random.sample(list(error_indices), sample_count)

        for idx in sample_indices:
            text = str(sample_texts[idx])
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"  '{text}'")
