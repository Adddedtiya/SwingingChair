import torch
import torch.nn             as nn
import torch.nn.functional  as F

from einops import rearrange, reduce, repeat, einsum

from .model_shared import *

class SimpleDecoder(nn.Module):
    def __init__(self, latent_size : int, output_channels : int):
        super().__init__()

        self.latent_size = latent_size
        self.fixed_size  = 4

        self.projected_size = self.latent_size * self.fixed_size * self.fixed_size

        self.project = nn.Sequential(
            nn.Linear(self.latent_size, self.projected_size),
            nn.LayerNorm(self.projected_size),
            nn.LeakyReLU()
        )

        self.extend = nn.Sequential(
            nn.Conv2d(self.latent_size, 512, kernel_size = 3, padding = 'same'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU()
        )

        self.body = nn.Sequential(
            
            # N, C, 4, 4
            ResidualGroupedBlock(512, 256, bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(256, 2),

            # N, C, 8, 8
            ResidualGroupedBlock(256, 128, bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(128, 2),

            # N, C, 16, 16
            ResidualGroupedBlock(128, 64,  bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(64, 2),
            
            # N, C, 32, 32
            ResidualGroupedBlock(64, 32,   bottleneck_ratios = [2, 4, 2]),
            UpShuffleLinear(32, 2),

            # N, C, 64, 64
        )

        # Trying with 1x1 conv, 
        self.head = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size = 1, padding = 'same')
        )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # non-linearly project the input into the output size
        x = self.project(x)

        # reshape the projected output
        x = rearrange(x, "n (c h w) -> n c h w", c = self.latent_size, h = self.fixed_size, w = self.fixed_size)

        # convolutional extrapolation
        x = self.extend(x)

        # body of the model
        x = self.body(x)

        # convolutional Head of the model
        x = self.head(x)
        
        return x



class SimpleEncoder(nn.Module):
    def __init__(self, input_channels : int, latent_size : int):
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size = 7, padding = 'same'),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU()
        )

        self.body = nn.Sequential(
            # N, C, 64, 64
            ResidualGroupedBlock(8, 16,    bottleneck_ratios = [2, 2, 2]),
            AvgMaxPool(16),
            
            # N, C, 32, 32
            ResidualGroupedBlock(16, 32,   bottleneck_ratios = [2, 2, 2]),
            AvgMaxPool(32),

            # N, C, 16, 16
            ResidualGroupedBlock(32, 64,   bottleneck_ratios = [2, 2, 2]),
            AvgMaxPool(64),

            # N, C,  8,  8
            ResidualGroupedBlock(64, 128,  bottleneck_ratios = [2, 2, 2]),
            AvgMaxPool(128),

            # N, C,  4,  4
            ResidualGroupedBlock(128, 256, bottleneck_ratios = [2, 2, 2]),
        ) 

        # project output head
        self.project_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 640),
            nn.LayerNorm(640),
            nn.LeakyReLU()
        )

        # project to latent space mu, log_var
        self.head_mu = nn.Linear(640, latent_size)
        self.head_lv = nn.Linear(640, latent_size)

    
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # basic forward pass
        x = self.feet(x)
        x = self.body(x)

        # project the X to flatten space
        x = rearrange(x, "n c h w -> n (c h w)")
        x = self.project_head(x)

        # compute mu, logvar
        x_mu = self.head_mu(x)
        x_lv = self.head_lv(x)

        return x_mu, x_lv
    

class SimpleCritic(nn.Module):
    def __init__(self, input_channels : int):
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size = 7, padding = 'same'),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU()
        )

        self.body = nn.Sequential(
            # N, C, 64, 64
            ConvInstance(8, 16),
            ResidualInstanceBlock(16),
            ResidualInstanceBlock(16),
            ResidualInstanceBlock(16),
            AvgMaxPool(16),
            
            # N, C, 32, 32
            ConvInstance(16, 32),
            ResidualInstanceBlock(32),
            ResidualInstanceBlock(32),
            ResidualInstanceBlock(32),
            AvgMaxPool(32),

            # N, C, 16, 16
            ConvInstance(32, 64),
            ResidualInstanceBlock(64),
            ResidualInstanceBlock(64), 
            AvgMaxPool(64),

            # N, C,  8,  8
            ConvInstance(64, 128),
            ResidualInstanceBlock(128),
            ResidualInstanceBlock(128),
            AvgMaxPool(128),

            # N, C,  4,  4
            ConvInstance(128, 256),
            ResidualInstanceBlock(256),
            ResidualInstanceBlock(256),
            AvgMaxPool(256),

            # N, C,  2,  2
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
        ) 

        self.head = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),

            nn.Linear(512, 1)
        )

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # standard model input
        x = self.feet(x)
        x = self.body(x)

        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.head(x)

        return x