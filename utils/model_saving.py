import torch
import sys
import os

# Set up root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_model import PneumoniaCNN  # Make sure this is the correct path to your model class

# Load the model
model = PneumoniaCNN()  # Instantiate the model
model.load_state_dict(torch.load('pneumonia_cnn_model.pth'))  # Load the model weights from file

# Save the model (weights or complete model)
torch.save(model.state_dict(), 'pneumonia_cnn_model_final.pth')  # Save just the weights

print("Model saved successfully!")
