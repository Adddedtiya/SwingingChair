import torch
import torch.nn as nn
from einops     import rearrange, reduce, repeat
import timm

class MoGrayDist(nn.Module):
    def __init__(self, grayscale : bool = True):
        super().__init__()

        # check if the input is grayscale
        self.grayscale = grayscale

        self.model = timm.create_model(
            'mobilenetv2_100.ra_in1k',
            pretrained  = True,
            num_classes = 0,  # remove classifier nn.Linear
        )

        self.head = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 1, kernel_size = 3, padding = 'same')
        )

    def preprocess(self, x : torch.Tensor) -> torch.Tensor:
        if self.grayscale:
            x = repeat(x, 'n c h w -> n (r c) h w', r = 3)
        return x

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        return self.model.forward_features(x)
    
    def forward_discriminator(self, x : torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        x = self.head(x)
        return x

    def forward_preceptual(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        y = self.forward(x)
        l = nn.functional.mse_loss(x, y)
        return l  



if __name__ == "__main__":

    model = MoGrayDist(True)
    model.eval()
    trand = torch.rand(1, 1, 512, 512)

    output = model.forward_discriminator(trand)
    print(output.shape) # N, 1, 16, 16
