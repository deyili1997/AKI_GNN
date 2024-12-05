from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd


def performance_per_class(y_test: np.array, y_test_pred: np.array, y_test_pred_prob: np.array) -> dict:
    """
    Computes performance metrics for binary classification focused on the positive class.

    Args:
        y_test (np.array): True labels (binary: 0 or 1).
        y_test_pred (np.array): Predicted labels (binary: 0 or 1).
        y_test_pred_prob (np.array): Predicted probabilities for the positive class (shape: [n_samples]).

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1 score, AUROC, and AUPRC for the positive class.
    """

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test, y_test_pred_prob)
    auprc = average_precision_score(y_test, y_test_pred_prob)

    # Return metrics in a dictionary
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUROC": auroc,
        "AUPRC": auprc
    }


def visualize_embeddings(h, color):
    # Perform t-SNE dimensionality reduction
    if isinstance(color, torch.Tensor):
        h = h.detach().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(h)
    
    # Ensure `color` is a CPU tensor or NumPy array
    if isinstance(color, torch.Tensor):
        color = color.cpu().numpy()
    
    # Define color mapping and labels
    color_mapping = ['green', 'yellow', 'orange', 'red']
    labels = ['AKI-0', 'AKI-1', 'AKI-2', 'AKI-3']
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    
    # Plot each category with its corresponding color and label
    for i, (c, label) in enumerate(zip(color_mapping, labels)):
        mask = (color == i)  # Select points corresponding to the current category
        plt.scatter(
            z[mask, 0], z[mask, 1], 
            s=1, c=c, label=label
        )
    
    # Add legend
    plt.legend(title="AKI Stages", loc='best', fontsize='large', title_fontsize='large')
    
    # Show plot
    plt.show()