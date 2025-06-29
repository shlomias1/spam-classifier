import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import os
from config import IMAGES_DIR
from utils.logger import _create_log
from processing import convert_binary_labels_to_minus_plus, convert_minus_plus_to_binary

def save_confusion_matrix(y_true, y_pred, labels, path):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap=plt.cm.Blues, values_format='d'
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_loss_trace(entropy_values, path, title):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    full_path = os.path.join(IMAGES_DIR, path)
    plt.figure(figsize=(10, 6))
    plt.plot(entropy_values, marker='o', linestyle='-', color='tab:blue', label='Entropy')
    min_idx = entropy_values.index(min(entropy_values))
    min_val = min(entropy_values)
    plt.scatter(min_idx, min_val, color='red', zorder=5, label=f'Min Entropy ({min_val:.2f})')
    plt.title(title)
    plt.xlabel("Split Step")
    plt.ylabel("loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_depth_sensitivity(results_array, path):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    full_path = os.path.join(IMAGES_DIR, path)
    plt.figure(figsize=(10, 6))
    plt.plot(results_array[:, 0], results_array[:, 1], label='Precision')
    plt.plot(results_array[:, 0], results_array[:, 2], label='Recall')
    plt.plot(results_array[:, 0], results_array[:, 3], label='F1 Score')
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs. Max Depth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_feature_importance(counter, feature_names, output_path, top_n=10):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    full_path = os.path.join(IMAGES_DIR, output_path)
    top_features  = counter.most_common(top_n)
    indices = [idx for idx, _ in top_features]
    labels = [feature_names[i] for i in indices]
    values = [count for _, count in top_features]
    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color='skyblue')
    plt.title("Top Feature Importances (by split count)")
    plt.xlabel("Split Count")
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
    
def plot_max_features_tuning(max_features_range, precisions, recalls, f1_scores, output_path):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    full_path = os.path.join(IMAGES_DIR, output_path)
    plt.figure(figsize=(10, 6))
    plt.plot(max_features_range, precisions, label='Precision', marker='o')
    plt.plot(max_features_range, recalls, label='Recall', marker='o')
    plt.plot(max_features_range, f1_scores, label='F1 Score', marker='o')
    plt.xlabel('Max Features')
    plt.ylabel('Score')
    plt.title('Random Forest Performance vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()

def plot_roc_curves_comparison(models_outputs, y_test, output_path = "roc_comparison.png"):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    full_path = os.path.join(IMAGES_DIR, output_path)
    plt.figure(figsize=(10, 8))
    for name, y_pred, y_score, label_type in models_outputs:
        if label_type == "plusminus":
            y_test_adj = convert_minus_plus_to_binary(y_test)
        else:
            y_test_adj = y_test
        fpr, tpr, _ = roc_curve(y_test_adj, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()
