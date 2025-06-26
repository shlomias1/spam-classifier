import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
from config import IMAGES_DIR

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