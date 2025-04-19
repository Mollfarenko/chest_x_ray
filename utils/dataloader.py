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
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizes to [-1, 1]
    ])

    # Load full train set from folder
    train_dataset_full = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)

    # Split into train and validation subsets
    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size
    train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size])

    # Load test set
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    # Create datasets dictionary (clear naming)
    datasets_dict = {
        "train": train_subset,
        "val_split": val_subset,
        "test": test_dataset
    }

    # Create loaders
    dataloaders = {
        key: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
        for key, dataset in datasets_dict.items()
    }

    return dataloaders, batch_size, image_size
