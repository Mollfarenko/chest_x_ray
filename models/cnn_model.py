import torch.nn as nn

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # grayscale images, 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),  # Assumes input images are resized to 128x128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
