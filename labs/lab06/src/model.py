"""CifarCNN model definition for CIFAR-10 classification."""

import torch
import torch.nn as nn


class CifarCNN(nn.Module):
    """Small CNN for CIFAR-10 classification."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 0
            nn.BatchNorm2d(32),                             # 1
            nn.ReLU(),                                      # 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 3
            nn.BatchNorm2d(64),                             # 4
            nn.ReLU(),                                      # 5
            nn.MaxPool2d(2),                                # 6
            nn.Dropout2d(0.25),                             # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8  <- last conv
            nn.BatchNorm2d(128),                            # 9
            nn.ReLU(),                                      # 10
            nn.MaxPool2d(2),                                # 11
            nn.Dropout2d(0.25),                             # 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
