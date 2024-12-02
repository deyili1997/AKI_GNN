from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd

def performance_per_class(y_test: np.array, y_test_pred: np.array, y_test_pred_prob: np.array) -> pd.DataFrame:
    
    n_class = len(np.unique(y_test))
    # Generate the classification report as a dictionary
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)

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