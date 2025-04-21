import os
import sys
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup path for imports ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_model import PneumoniaCNN
from utils.dataloader import get_dataloaders


def load_trained_model(path, device):
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.view(-1, 1).cpu().numpy())

    all_preds = [int(p[0]) for p in all_preds]
    all_labels = [int(l[0]) for l in all_labels]

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_improved": cm,
        "predictions": all_preds,
        "true_labels": all_labels
    }


def log_test_results(metrics, model, run_name="Test Evaluation"):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("phase", "test")

        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        mlflow.log_metric("test_precision", metrics["precision"])
        mlflow.log_metric("test_recall", metrics["recall"])
        mlflow.log_metric("test_f1_score", metrics["f1_score"])

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            metrics["confusion_matrix_improved"],
            annot=True,
            fmt='d',
            cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        fig_path = "reports/confusion_matrix_improved.png"
        plt.savefig(fig_path)
        mlflow.log_artifact(fig_path)

        # Save model
        mlflow.pytorch.log_model(model, "model_test_eval")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_trained_model("models/pneumonia_model.pth", device)

    dataloaders, _, _ = get_dataloaders("data", batch_size=64)
    test_loader = dataloaders["test"]

    metrics = evaluate_model(model, test_loader, device)

    # Print results
    print("\nTest Metrics:")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key.capitalize():<10}: {metrics[key]:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix_improved']}")

    log_test_results(metrics, model)


if __name__ == "__main__":
    main()
