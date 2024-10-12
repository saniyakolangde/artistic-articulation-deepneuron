import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    '''

        So CycleGan uses a network of stacked residual blocks, which are essentially two 3x3 convolutional layers
        after each other.

        Also note that unlike the discriminator, the generator for CycleGan both upsamples and downsamples,
        so we must take this into account.

    '''
    def __init__(self, in_channels, out_channels, downsample_flag=True, use_relu=True, **kwargs):
        super().__init__()
        if downsample_flag:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs), # if downsampling, use regular convolutions
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_relu else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs), # This is for the fractional-strided convolution noted in CycleGan paper
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_relu else nn.Identity(),
            )

    def forward(self, input):
        return self.conv(input)

class ResidualBlock(nn.Module): # we define a different class object for our residual block
    def __init__(self, in_channels):
        super().__init__()
        self.residual = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1), # kwargs is kernel_size and padding
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, use_relu=False), # residual block has no change in dimensionality!
        )

    def forward(self, input):
        return input + self.residual(input) # residual/skip connection

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_dims = [64, 128, 256], num_residuals=9): # note the number of residual blocks depends on input image dimensions, use 6 for 128x128
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(in_channels, hidden_dims[0], kernel_size=7, stride=1, padding=3), # initial layer
            ConvBlock(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1), # downsample block 1
            ConvBlock(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1), # downsample block 2
            *[ResidualBlock(hidden_dims[2]) for _ in range(num_residuals)], # basically create num_residuals amount of 256-residual blocks
            ConvBlock(hidden_dims[2], hidden_dims[1], downsample_flag=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1),  # upsample block 1
            ConvBlock(hidden_dims[1], hidden_dims[0], downsample_flag=False, kernel_size=3, stride=2, padding=1,
                      output_padding=1),  # upsample block 2
            nn.Conv2d(hidden_dims[0], in_channels, kernel_size=7,stride=1, padding=3, padding_mode="reflect"),
        )

    def forward(self, input):
        return torch.tanh(self.model(input)) # wonder why they use tanh for -1,1 range

