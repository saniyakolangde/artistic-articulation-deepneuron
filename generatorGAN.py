import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, seed_size):
        super().__init__()
        self.model = nn.Sequential(
            # Input seed_size x 1 x 1
            nn.ConvTranspose2d(seed_size, 128, kernel_size=4, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer output: 256 x 4 x 4

            nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer output: 128 x 8 x 8

            nn.ConvTranspose2d(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer output: 64 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Layer output: 32 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, padding=1, stride=2, bias=False),
            nn.Tanh()
            # Output: 3 x 64 x 64
        )

    def forward(self, x):
        return torch.tanh(self.model(x))
