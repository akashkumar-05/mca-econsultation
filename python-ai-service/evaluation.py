"""
Model Evaluation Module for MCA eConsultation AI Service.

Provides a complete evaluation pipeline for the sentiment classifier:
  - Accuracy, Precision, Recall, F1-score (macro & weighted)
  - Confusion matrix generation and visualization
  - Per-class performance breakdown
  - Evaluation report saved to disk for reproducibility

This module is critical for research validation and conference papers.
"""

import os
import json
import logging
import time
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_sentiment_model(model, tokenizer, texts: list, labels: list,
                             label_map: dict, device, output_dir: str,
                             batch_size: int = 32, max_length: int = 256) -> dict:
    """
    Run full evaluation pipeline on the sentiment model.

    Args:
        model: The loaded sentiment classification model
        tokenizer: The corresponding tokenizer
        texts: List of input text strings
        labels: List of ground-truth integer labels (0, 1, 2)
        label_map: Dict mapping int → string label
        device: torch device (cpu/cuda/mps)
        output_dir: Directory to save evaluation artifacts
        batch_size: Inference batch size
        max_length: Max token length

    Returns:
        Dictionary containing all evaluation metrics
    """
    import torch
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )

    logger.info(f"Starting evaluation on {len(texts)} samples...")
    start_time = time.time()

    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    # Batch inference
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    elapsed = time.time() - start_time
    labels_np = np.array(labels)
    preds_np = np.array(all_preds)

    # ── Compute Metrics ──
    label_names = [label_map[i] for i in sorted(label_map.keys())]

    accuracy = accuracy_score(labels_np, preds_np)
    precision_macro = precision_score(labels_np, preds_np, average="macro", zero_division=0)
    recall_macro = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average="macro", zero_division=0)
    f1_weighted = f1_score(labels_np, preds_np, average="weighted", zero_division=0)

    cm = confusion_matrix(labels_np, preds_np)
    class_report = classification_report(
        labels_np, preds_np, target_names=label_names, output_dict=True, zero_division=0
    )

    # ── Build Results ──
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(texts),
        "inference_time_seconds": round(elapsed, 3),
        "throughput_samples_per_sec": round(len(texts) / elapsed, 1) if elapsed > 0 else 0,
        "device": str(device),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
    }

    for name in label_names:
        if name in class_report:
            results["per_class"][name] = {
                "precision": round(class_report[name]["precision"], 4),
                "recall": round(class_report[name]["recall"], 4),
                "f1": round(class_report[name]["f1-score"], 4),
                "support": int(class_report[name]["support"]),
            }

    # ── Save to disk ──
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation report saved to: {report_path}")

    # ── Save confusion matrix as image ──
    try:
        _save_confusion_matrix_plot(cm, label_names, output_dir)
    except Exception as e:
        logger.warning(f"Could not save confusion matrix plot: {e}")

    # ── Log summary ──
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Samples:    {len(texts)}")
    logger.info(f"  Accuracy:   {accuracy:.4f}")
    logger.info(f"  Precision:  {precision_macro:.4f} (macro)")
    logger.info(f"  Recall:     {recall_macro:.4f} (macro)")
    logger.info(f"  F1-score:   {f1_macro:.4f} (macro)")
    logger.info(f"  F1-score:   {f1_weighted:.4f} (weighted)")
    logger.info(f"  Throughput: {results['throughput_samples_per_sec']} samples/sec")
    logger.info(f"  Device:     {device}")
    logger.info("=" * 60)

    return results


def _save_confusion_matrix_plot(cm, labels, output_dir):
    """Save confusion matrix as a PNG image using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="Confusion Matrix — Sentiment Classification",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to: {path}")
