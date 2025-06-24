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
    path = os.path.join(IMAGES_DIR, path)    
    plt.figure()
    plt.plot(entropy_values, marker='o')
    plt.title(title)
    plt.xlabel("Split Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()