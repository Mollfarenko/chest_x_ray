import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, image_size=224, batch_size=128, num_workers=16):
    """
    Creates PyTorch dataloaders for train, validation, and test sets.

    Args:
        data_dir (str): Path to the root data folder containing 'train', 'val', 'test' subfolders.
        image_size (int): Size to resize images to (default 224x224).
        batch_size (int): Batch size for DataLoaders.

    Returns:
        dict: A dictionary with keys 'train', 'val', 'test' and DataLoader objects as values.
    """

    # Define transform: Resize + Normalize pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # Converts [0, 255] → [0.0, 1.0]
    ])

    # Create datasets from folders
    data = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
        for split in ["train", "val", "test"]
    }

    # Create loaders
    dataloaders = {
        split: DataLoader(
            data[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True  # Important for faster CPU → GPU transfer
        )
        for split in ["train", "val", "test"]
    }

    return dataloaders, batch_size, image_size
