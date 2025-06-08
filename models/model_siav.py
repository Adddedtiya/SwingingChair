import torch
import torch.nn             as nn
import torch.nn.functional  as F

from einops import rearrange, reduce, repeat, einsum


### Machine Learning Models ###

class LightweightDecoder(nn.Module):
    def __init__(self, latent_space : int, output_channels : int):
        super().__init__()

        self.linear_projection = nn.Sequential(
            nn.Linear(latent_space, 512 * 4 * 4),
            nn.LeakyReLU()
        )

        self.convolutional_layers = nn.Sequential(

            # N, 512, 4, 4
            SimpleResConv(512),
            SimpleResConv(512),
            UpConvShuffle(512, 256, 2),

            # N, 256, 8, 8
            SimpleResConv(256),
            SimpleResConv(256),
            SimpleResConv(256),
            UpConvShuffle(256, 128, 2),

            # N, 128, 16, 16
            SimpleResConv(128),
            SimpleResConv(128),
            SimpleResConv(128),
            UpConvShuffle(128, 64, 2),

            # N,  64, 32, 32
            SimpleResConv(64),
            SimpleResConv(64),
            UpConvShuffle(64, 32, 2),

            # N,  32, 64, 64
            SimpleResConv(32),
            SimpleResConv(32)
        )


        self.head = nn.Sequential(
            nn.Conv2d(32, output_channels, kernel_size = 3, padding = 'same', bias = True),
            nn.Sigmoid() 
        )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # linear projection
        x = self.linear_projection(x)
        x = rearrange(x, 'n (c h w) -> n c h w', c = 512, h = 4, w = 4)

        # feature generations
        x = self.convolutional_layers(x)

        # output projection
        x = self.head(x)

        return x


class LightweightEncoder(nn.Module):
    def __init__(self, input_channels : int, latent_space : int):
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.convolutional_layers = nn.Sequential(
            
            # N,  32, 64, 64
            SimpleResConv(32),
            SimpleResConv(32),
            DownAvgMaxPool(32, 64),

            # N,  64, 32, 32
            SimpleResConv(64),
            SimpleResConv(64),
            DownAvgMaxPool(64, 128),

            # N, 128, 16, 16
            SimpleResConv(128),
            SimpleResConv(128),
            SimpleResConv(128),
            DownAvgMaxPool(128, 256),

            # N, 256, 8, 8
            SimpleResConv(256),
            SimpleResConv(256),
            SimpleResConv(256),
            DownAvgMaxPool(256, 512),

            # N, 512, 4, 4
            SimpleResConv(512),
            SimpleResConv(512),       
        )

        self.output_projection = nn.Sequential(
            nn.Linear(512 * 4 * 4, latent_space),
            nn.Tanh()
        )       


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # input projection
        x = self.feet(x)

        # feature compression
        x = self.convolutional_layers(x)
        
        # linear projection
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.output_projection(x)

        return x


class LigweightAutoencoder(nn.Module):
    def __init__(self, input_channels : int, latent_size : int, output_channels : int):
        super().__init__()

        self.encoder = LightweightEncoder(input_channels, latent_size)
        self.decoder = LightweightDecoder(latent_size, output_channels)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_encoder(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forwar_decoder(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


### Functional Blocks ###

class UpConvShuffle(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, ratio : int, kernel : int = 3):
        super().__init__()

        projected_channels = output_channels * (ratio * ratio)
        self.conv = nn.Conv2d(input_channels, projected_channels, kernel, padding = 'same', bias = True)

        self.shuffle = nn.PixelShuffle(ratio)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # linearly project input
        x = self.conv(x)

        # shuffle to resize
        x = self.shuffle(x)

        return x

class DownAvgMaxPool(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, kernel : int = 3):
        super().__init__()

        # Average And Max Pooling
        self.max_pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)

        # linearlly project back 
        self.project = nn.Conv2d(input_channels * 2, output_channels, kernel_size = kernel, padding = 'same')
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        mx = self.max_pool(x)
        ax = self.avg_pool(x)

        # concat both in channel dimension
        cx = torch.cat([mx, ax], dim = 1)

        # project back to the original shape
        ox = self.project(cx)
        return ox  

class SimpleResConv(nn.Module):
    def __init__(self, channels : int, kernel_size = 3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding = 'same', bias = False),
            nn.BatchNorm2d(channels, momentum = 0.5),
            nn.LeakyReLU(inplace = False),

            nn.Conv2d(channels, channels, kernel_size, padding = 'same', bias = False),
            nn.BatchNorm2d(channels, momentum = 0.5),
            nn.LeakyReLU(inplace = False)
        )

    def forward(self, x):
        out = self.block(x)
        out = out + x
        return out
    



if __name__ == "__main__":
    print("SIAV")

    m = LigweightAutoencoder(1, 512, 1)
    m.eval()

    t = torch.rand(1, 1, 64, 64)

    from torchinfo import summary
    summary(m, input_data = t)

    print(m)