import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch

from utils.dataloader import get_dataloaders
from models.cnn_model import PneumoniaCNN
from utils.save_results import save_training_results
from utils.training_loop import train_model
from utils.plots import plot_training_history

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = "data"
    dataloaders, batch_size, image_size = get_dataloaders(data_dir)

    # Model
    model = PneumoniaCNN().to(device)

    # Loss and optimizer
    learning_rate = 0.001
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 25
    patience = 5

    with mlflow.start_run(run_name="Pneumonia-CNN"):
        model, history, final_metrics = train_model(model, dataloaders, criterion, optimizer, num_epochs, device, patience)

        save_training_results(
            history=history,
            final_metrics={
                "val_acc": final_metrics["val_acc"],
                "val_f1": final_metrics["val_f1"],
                "model_state_dict": model.state_dict()
            },
            model_path="models/pneumonia_model.pth",
            history_path="reports/training_history.json",
            report_path="reports/final_metrics.txt"
        )

        # Log to MLflow
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", dataloaders["train"].batch_size)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("criterion", criterion.__class__.__name__)
        mlflow.log_param("patience", patience)

        mlflow.log_metric("val_accuracy", final_metrics["val_acc"])
        mlflow.log_metric("val_f1_score", final_metrics["val_f1"])

        mlflow.log_artifact("reports/training_history.json")
        mlflow.log_artifact("reports/final_metrics.txt")

        mlflow.pytorch.log_model(model, "model")


        # After training is complete
        plot_training_history(
            history_path="reports/training_history.json",
            save_path="reports/training_plot.png"
        )


if __name__ == "__main__":
    main()
