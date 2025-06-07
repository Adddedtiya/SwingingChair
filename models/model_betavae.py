import torch
import torch.nn             as nn
import torch.nn.functional  as F

from einops import rearrange, reduce, repeat, einsum

from .model_shared import *


class DecoderFeetFixed(nn.Module):
    def __init__(self, latent_size : int, output_channels : int):
        super().__init__()

        self.latent_size      = latent_size
        self.output_channels  = output_channels
        self.fixed_size       = 4 # (N, C, 4, 4)
        self.initial_channels = 256
        
        # computed
        projected_size = self.initial_channels * (self.fixed_size * self.fixed_size)
        self.project = nn.Sequential(
            nn.Linear(latent_size, projected_size),
            nn.LayerNorm(projected_size),
            nn.LeakyReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.initial_channels, self.output_channels, 3, padding = 'same'),
            nn.InstanceNorm2d(self.output_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # non-linearly project the input into the output size
        x = self.project(x)

        # reshape the projected output
        x = rearrange(x, "n (c h w) -> n c h w", c = self.initial_channels, h = self.fixed_size, w = self.fixed_size)

        # convolutional extrapolation
        x = self.conv(x)

        return x
    

class FixedResidualDecoder(nn.Module):
    def __init__(self, latent_size : int, output_channels : int):
        super().__init__()

        self.feet = DecoderFeetFixed(latent_size, 1024)

        self.body = nn.Sequential(
            
            # N, C, 4, 4
            ResidualGroupedBlock(1024, 512, bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(512, 2),

            # N, C, 8, 8
            ResidualGroupedBlock(512, 256, bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(256, 2),

            # N, C, 16, 16
            ResidualGroupedBlock(256, 128, bottleneck_ratios = [2, 4, 4, 2]),
            UpShuffleLinear(128, 2),
            
            # N, C, 32, 32
            ResidualGroupedBlock(128, 64,  bottleneck_ratios = [2, 4, 4, 2]),
            UpShuffleLinear(64, 2),

            # N, C, 64, 64
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size = 3, padding = 'same')
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feet(x)
        x = self.body(x)
        x = self.head(x)
        return x











