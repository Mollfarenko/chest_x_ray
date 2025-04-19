import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_mean_std(data_dir, image_size=224, batch_size=32):
    # Only basic transforms: resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_images = 0

    print("Computing mean and std over training data...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)  # batch size (N)
        images = images.view(batch_samples, -1)  # Flatten images to (N, H*W)
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.item(), std.item()

# Example usage
if __name__ == "__main__":
    data_dir = "../data"  # Adjust path if needed
    mean, std = compute_mean_std(data_dir)
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
