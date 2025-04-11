import sys
import os

# Add the root directory of the project to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataloader import get_dataloaders

# Path to your dataset folder
data_dir = "data"

# Load dataloaders
dataloaders = get_dataloaders(data_dir)

# Count the number of samples in each loader
for split in ["train", "val", "test"]:
    loader = dataloaders[split]
    dataset_size = len(loader.dataset)
    print(f"{split.capitalize()} set: {dataset_size} images")

