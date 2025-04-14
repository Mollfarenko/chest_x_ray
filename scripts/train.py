import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch


def main():
    # Set up root path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from utils.dataloader import get_dataloaders
    from models.cnn_model import PneumoniaCNN

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # Load data
    data_dir = "data"
    dataloaders, batch_size, image_size = get_dataloaders(data_dir)

    # Model
    model = PneumoniaCNN()
    model.to(device)

    # Loss and optimizer
    learning_rate = 0.001
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10  # Start small while testing

    mlflow.set_experiment("Pneumonia_CNN_Classification")
    # Start MLflow run
    with mlflow.start_run(run_name="Run_with_BS128_LR0.001_EP10"):

        # Log parameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("image_size", image_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)


        for epoch in range(num_epochs):
            model.train()  # Set to training mode
            running_loss = 0.0

            for features, labels in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}"):
                features = features.to(device)
                labels = labels.view(-1, 1).float().to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

            avg_loss = running_loss / len(dataloaders["train"])
            print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}")

            # Log metric for this epoch
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Save the model
        mlflow.pytorch.log_model(model, "pneumonia_cnn_model")

        # Save only weights
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/pneumonia_cnn_weights.pth")

        pass

if __name__ == '__main__':
    main()
