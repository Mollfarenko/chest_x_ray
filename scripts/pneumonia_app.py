import os
import sys
import torch
from torchvision import transforms
from PIL import Image

# Set up root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cnn_model import PneumoniaCNN  # Modify if necessary to match your model's location


def load_model(model_path='models/pneumonia_cnn_weights.pth'):
    """ Load the trained model weights """
    device = torch.device("cpu")
    model = PneumoniaCNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, device

def preprocess_image(image_path, input_size=(224, 224)):
    """ Preprocess the image to be ready for the model """
    transform = transforms.Compose([
        transforms.Resize(input_size),            # Resize image to match model's input size
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (if needed)
        transforms.ToTensor(),                    # Convert image to PyTorch tensor
    ])

    image = Image.open(image_path).convert('RGB')  # Open image
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, device, image_path):
    """ Predict if the image has pneumonia or not """
    image = preprocess_image(image_path)  # Preprocess the image
    image = image.to(device)  # Move the image to the right device

    with torch.no_grad():  # Disable gradients for inference
        output = model(image)
        prediction = torch.round(output)  # Get the final prediction (0 or 1)

    return prediction.item()  # Return the predicted label (0 or 1)

def main():
    print("Welcome to the Pneumonia Detection App!")

    # Load the model
    model, device = load_model()

    # Ask the user to input the image path
    image_path = input("Please enter the path to the X-ray image: ")

    if not os.path.exists(image_path):
        print("Error: Image not found. Please check the path and try again.")
        return

    # Get the prediction
    prediction = predict_image(model, device, image_path)

    # Output the result
    if prediction == 0:
        print("The model predicts: NORMAL")
    else:
        print("The model predicts: PNEUMONIA")

if __name__ == "__main__":
    main()
