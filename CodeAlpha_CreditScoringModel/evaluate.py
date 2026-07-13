"""
evaluate.py
Location: CodeAlpha_CreditScoringModel/evaluate.py

Step 4: Evaluate models with Precision, Recall, F1, ROC-AUC,
and plot ROC curves + feature importance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-Score :", f1_score(y_true, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print()


def plot_roc_curves(y_test, proba_dict, save_path="outputs/roc_curve.png"):
    plt.figure(figsize=(7, 6))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curve plot to {save_path}")


def plot_feature_importance(model, feature_names, save_path="outputs/feature_importance.png"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances.sort_values(ascending=False).head(10).plot(kind='barh')
    plt.title("Top 10 Important Features")
    plt.gca().invert_yaxis()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to {save_path}")
