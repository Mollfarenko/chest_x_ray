import os
import sys
import torch
from torchsummary import summary
from contextlib import redirect_stdout

def main():
    # Set up path to import model
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.cnn_model import PneumoniaCNN

    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN().to(device)

    # Print model summary
    summary(model, input_size=(1, 224, 224))  # Adjust to your image input size

    # Create dummy input and move to device
    dummy_input = torch.randn(1, 1, 224, 224).to(device)

    # Ensure reports folder exists
    export_path = os.path.join("reports", "pneumonia_model.onnx")
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    # Export to ONNX
    torch.onnx.export(model, dummy_input, export_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11)

    print(f"Model exported to {export_path}")

    with open("reports/model_summary.txt", "w") as f:
        with redirect_stdout(f):
            summary(model, input_size=(1, 224, 224))


if __name__ == '__main__':
    main()


