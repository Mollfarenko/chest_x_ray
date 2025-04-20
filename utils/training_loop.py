import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.early_stopping import EarlyStopping


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, device="cuda", patience=5):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print("Using device:", device)
    if device == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 30)

        # --- Training ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in tqdm(dataloaders["train"], desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.view(-1, 1).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(torch.round(outputs).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        train_loss /= len(dataloaders["train"].dataset)
        train_acc = (torch.tensor(train_preds) == torch.tensor(train_labels)).float().mean().item()
        train_f1 = f1_score(train_labels, train_preds)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)

        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1 Score: {train_f1:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloaders["val_split"], desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                labels = labels.view(-1, 1).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(torch.round(outputs).detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

        val_loss /= len(dataloaders["val_split"].dataset)
        val_acc = (torch.tensor(val_preds) == torch.tensor(val_labels)).float().mean().item()
        val_f1 = f1_score(val_labels, val_preds)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1 Score: {val_f1:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(early_stopping.best_model.state_dict())

    final_metrics = {
        "val_acc": val_acc,
        "val_f1": val_f1
    }

    return model, history, final_metrics
