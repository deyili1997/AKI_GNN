from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def performance_per_class(y_test: np.array, y_test_pred: np.array, y_test_pred_prob: np.array) -> pd.DataFrame:
    
    n_class = len(np.unique(y_test))
    # Generate the classification report as a dictionary
    # address the problem of zero_division
    
    report_dict = classification_report(y_test, y_test_pred, output_dict=True, zero_division = 0)

    # Convert to a pandas DataFrame for better formatting
    metrics_table = pd.DataFrame(report_dict).transpose()

    # Extract only the classes (0, 1, 2, 3) and metrics
    class_metrics = metrics_table.loc[[str(c) for c in range(n_class)], ['precision', 'recall', 'f1-score']]
    
    # add AUROC and AUPRC
    AUROC = []
    AUPRC = []
    for i in range(n_class):
        y_true_binary = (y_test == i).astype(int)
        y_proba_class = y_test_pred_prob[:, i]
        # compute AUROC
        AUROC.append(roc_auc_score(y_true_binary, y_proba_class))
        # compute AUPRC
        AUPRC.append(average_precision_score(y_true_binary, y_proba_class))
        
    # add to the class_metrics DataFrame
    class_metrics['AUROC'] = AUROC
    class_metrics['AUPRC'] = AUPRC
    return class_metrics


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