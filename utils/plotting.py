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