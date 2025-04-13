import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def main():
    # Set up root path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from utils.dataloader import get_dataloaders
    from models.cnn_model import PneumoniaCNN

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)
    if device.type == "cuda":
        print("🧠 GPU name:", torch.cuda.get_device_name(0))

    # Load data
    data_dir = "data"
    dataloaders = get_dataloaders(data_dir)

    # Model
    model = PneumoniaCNN()
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10  # Start small while testing

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

    # Save only weights
    torch.save(model.state_dict(), "models/pneumonia_cnn_weights.pth")
    pass

if __name__ == '__main__':
    main()
