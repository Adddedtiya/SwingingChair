import torch
import torch.nn             as nn
import torch.nn.functional  as F

from einops import rearrange, reduce, repeat, einsum


class UpShuffleLinear(nn.Module):
    def __init__(self, channels : int, ratio : int):
        super().__init__()

        projected_channels = channels * (ratio * ratio)
        self.conv = nn.Conv2d(channels, projected_channels, 1, padding = 'same')

        self.shuffle = nn.PixelShuffle(ratio)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # linearly project input
        x = self.conv(x)

        # shuffle to resize
        x = self.shuffle(x)

        return x

# I dont really know why i have this honestly, seems not really usefull
class DownShuffleLinear(nn.Module):
    def __init__(self, channels : int, ratio : int):
        super().__init__()

        self.shuffle = nn.PixelUnshuffle(ratio)

        projected_channels = channels * (ratio * ratio)
        self.conv = nn.Conv2d(projected_channels, channels, 1, padding = 'same')
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # shuffle to resize
        x = self.shuffle(x)

        # linearly project output
        x = self.conv(x)

        return x


class BottleneckResidual(nn.Module):
    def __init__(self, channels : int, ratio : int = 4):
        super().__init__()

        hidden_size = (channels // ratio)

        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden_size, 1, padding = 'same'),
            nn.InstanceNorm2d(hidden_size),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_size, hidden_size, 3, padding = 'same'),
            nn.InstanceNorm2d(hidden_size),
            nn.LeakyReLU()
        )

        # project back with linear mapping
        self.project = nn.Conv2d(hidden_size, channels, 1, padding = 'same')
        nn.init.zeros_(self.project.weight)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # compute the residual bottleneck
        bottle = self.block(x)
        bottle = self.project(bottle)

        # add the residual
        x = x + bottle
        return x


class ResidualGroupedBlock(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, bottleneck_ratios : list[int] = [2, 2]):
        super().__init__()

        self.feet_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding = 'same'),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU()
        )

        bottles : list[BottleneckResidual] = []
        for ratio_value in bottleneck_ratios:
            bottles.append(
                BottleneckResidual(output_channels, ratio_value)
            )
        self.bottleneck_residual = nn.Sequential(*bottles)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feet_block(x)
        x = self.bottleneck_residual(x)
        return x
    


class AvgMaxPool(nn.Module):
    def __init__(self, channels : int, kernel_size : int = 2, stride : int = 2):
        super().__init__()

        # Average And Max Pooling
        self.max_pool = nn.MaxPool2d(kernel_size, stride)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride)

        # linearlly project back 
        self.project = nn.Conv2d(channels * 2, channels, kernel_size = 1, padding = 'same')
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mx = self.max_pool(x)
        ax = self.avg_pool(x)

        # concat both in channel dimension
        cx = torch.cat([mx, ax], dim = 1)

        # project back to the original shape
        ox = self.project(cx)
        return ox


class ConvInstance(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel_size : int = 3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding = 'same', bias = False),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace = False)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualInstanceBlock(nn.Module):
    def __init__(self, channels : int, kernel_size = 3):
        super().__init__()

        self.conv_block = nn.Sequential(
            ConvInstance(channels, channels, kernel_size),
            ConvInstance(channels, channels, kernel_size)
        )

    def forward(self, x):
        o = self.conv_block(x)
        o = o + x
        return o
