from utils.dataloader import get_dataloaders

# Path to your dataset folder
data_dir = "../data"

dataloaders = get_dataloaders(data_dir)

train_loader = dataloaders["train"]
val_loader = dataloaders["val"]
test_loader = dataloaders["test"]
