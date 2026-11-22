"""evaluate.py - hàm đánh giá dùng chung cho cả nhóm (TF-IDF, BiLSTM, PhoBERT).

Cách dùng:
    from evaluate import evaluate, summarize

    evaluate(y_true, y_pred, model_name="tfidf_lr_segmented_uni", save_dir="results")
    summarize("results")   # in bảng so sánh tất cả mô hình đã lưu

Nhãn: 0 = negative, 1 = neutral, 2 = positive.
Chỉ số chính của đề tài: macro-F1 (vì lớp neutral chỉ chiếm ~4-5%).
"""
import glob
import json
import os

import matplotlib

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABELS = [0, 1, 2]
LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred):
    """Trả về dict: accuracy, macro_f1, weighted_f1 và F1 từng lớp."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1_each = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    for name, value in zip(LABEL_NAMES, f1_each):
        metrics[f"f1_{name}"] = float(value)
    return metrics


def evaluate(y_true, y_pred, model_name="model", save_dir=None, show_plot=True):
    """Đánh giá một mô hình.

    y_true, y_pred : nhãn thật và nhãn dự đoán (list/array/Series, giá trị 0/1/2)
    model_name     : tên mô hình, dùng làm tiêu đề và tên file lưu
    save_dir       : nếu có, lưu <model_name>_metrics.json và <model_name>_cm.png
    show_plot      : False nếu chạy trong script không có màn hình
    """
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    metrics = compute_metrics(y_true, y_pred)

    print(f"===== {model_name} =====")
    print(
        classification_report(
            y_true, y_pred, labels=LABELS, target_names=LABEL_NAMES, digits=4, zero_division=0
        )
    )
    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        cmap="Blues", ax=ax, values_format="d"
    )
    ax.set_title(f"Confusion matrix - {model_name}")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        fig.savefig(os.path.join(save_dir, f"{model_name}_cm.png"), dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return metrics


def summarize(save_dir="results"):
    """Gom mọi file *_metrics.json trong save_dir thành một bảng (DataFrame), sắp theo macro-F1."""
    import pandas as pd

    rows = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*_metrics.json"))):
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        row["model"] = os.path.basename(path).replace("_metrics.json", "")
        rows.append(row)
    if not rows:
        print(f"Chưa có kết quả nào trong '{save_dir}'.")
        return None
    df = pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)
    print(df.round(4).to_string())
    return df
