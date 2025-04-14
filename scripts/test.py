import os
import sys
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch

def main():

    # Set up path to import model and dataloader
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.cnn_model import PneumoniaCNN
    from utils.dataloader import get_dataloaders

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("models/pneumonia_cnn_weights.pth", map_location=device))
    model.eval()

    # Load test dataloader
    data_dir = "data"
    dataloaders, batch_size, image_size = get_dataloaders(data_dir, batch_size=64)
    test_loader = dataloaders["test"]

    # Prepare metric accumulators
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.view(-1, 1).cpu().numpy())

    # Convert to flat arrays
    all_preds = [int(p[0]) for p in all_preds]
    all_labels = [int(l[0]) for l in all_labels]

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Print
    print("\nTest Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # === Log with MLflow ===
    with mlflow.start_run(run_name="Test Evaluation"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Optional: log confusion matrix as artifact
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        # Log the model (again, if needed)
        mlflow.pytorch.log_model(model, "model_test_eval")

    pass

if __name__ == '__main__':
    main()
