import torch
import torch.nn             as nn
import torch.nn.functional  as F

from einops import rearrange, reduce, repeat, einsum

# pip install vector-quantize-pytorch
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

### Machine Learning Models ###

class LightweightDecoderV7(nn.Module):
    def __init__(self, output_channels : int):
        super().__init__()

        self.conv_project = nn.Sequential(
            nn.Conv2d(8, 256, kernel_size = 3, padding = 'same', bias = True),
            nn.LeakyReLU()
        )

        self.convolutional_layers = nn.Sequential(

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
            SimpleResConv(64),
            UpConvShuffle(64, 32, 2),

            # N,  32, 64, 64
            SimpleResConv(32),
            SimpleResConv(32),
            UpConvShuffle(32, 16, 2),

            # N,  16, 128, 128
            SimpleResConv(16),
            SimpleResConv(16),
        )


        self.head = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size = 3, padding = 'same', bias = True),
            nn.Sigmoid() 
        )


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # projection
        x = self.conv_project(x)

        # feature generations
        x = self.convolutional_layers(x)

        # output projection
        x = self.head(x)

        return x

    def forward_head_conv(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # projection
        x_project = self.conv_project(x)

        # feature generations
        x_conv_out = self.convolutional_layers(x_project)

        # output projection
        x_final_out = self.head(x_conv_out)

        return (x_conv_out, x_final_out)


class LightweightEncoderV7(nn.Module):
    def __init__(self, input_channels : int):
        super().__init__()

        self.feet = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size = 3, padding = 'same', bias = True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        self.convolutional_layers = nn.Sequential(
            
            # N,  16, 128, 128
            SimpleResConv(16),
            SimpleResConv(16),
            DownAvgMaxPool(16, 32),

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
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size = 3, padding = 'same', bias = True)
        )       


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # input projection
        x = self.feet(x)

        # feature compression
        x = self.convolutional_layers(x)
        
        # projection
        x = self.output_projection(x)

        return x


class LigweightAutoencoderRK512(nn.Module):
    def __init__(self, input_channels : int, output_channels : int):
        super().__init__()

        self.encoder = LightweightEncoderV7(input_channels)
        self.decoder = LightweightDecoderV7(output_channels)

        self.quantizer = ResidualVQ(
            dim               = 8,
            codebook_size     = 256,
            codebook_dim      = 8,
            num_quantizers    = 8,
            use_cosine_sim    = True,
            accept_image_fmap = True,
            stochastic_sample_codes = True,
            sample_codebook_temp    = 0.1, # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook         = True # whether to share the codebooks for all quantizers or not
        )
        
    def freeze_anything_but_head(self) -> None:
        # try to freez the param
        for param_name, param_object in self.named_parameters():
            if param_name.startswith("decoder.convolutional_layers.15"):
                break

            # freeze until decoder conv layer 15
            param_object.requires_grad = False
        print("! Anything but the few head layers are frozen !")

    def freeze_layers_except(self, layers : list[str]) -> None:
        # freeze layer but the selected one
        for param_name, param_object in self.named_parameters():
            if any(param_name.startswith(x) for x in layers):
                continue

            param_object.requires_grad = False

        print(f"! Anything but {layers} layers are frozen !")

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)

        # compute the codebook first
        quantized, _, _ = self.quantizer(x)

        x = self.decoder(quantized)
        return x
    
    def forward_with_loss(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        # compute the codebook first
        quantized, _, entropy_aux_loss = self.quantizer(x)

        x = self.decoder(quantized)
        return x, entropy_aux_loss

    def forward_encoder(self, x : torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        q_x, _, _ = self.quantizer(x)
        return q_x
    
    def forwar_decoder(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def forward_decoder_tuple(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder.forward_head_conv(x)

    def forward_encoder_tokens_targets(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compressed_latent = self.encoder(x)
        z_quant, indc_tensor, _ = self.quantizer(compressed_latent)

        # flatten indicies tensors
        flatten_indicies : torch.Tensor = rearrange(indc_tensor, 'n h w -> n (h w)')
        flatten_indicies : torch.Tensor = flatten_indicies.to(torch.long)
        
        # convert to one hot indicies
        onehot_indicies  : torch.Tensor = nn.functional.one_hot(flatten_indicies, num_classes = 256).to(torch.float32)

        # flatten z space
        flatten_z_quant : torch.Tensor = rearrange(z_quant, 'n c h w -> n (h w) c')
        flatten_z_quant : torch.Tensor = flatten_z_quant.detach()

        return (flatten_z_quant, flatten_indicies, onehot_indicies)

    def forward_decoder_tokens(self, x : torch.Tensor) -> torch.Tensor:

        # reshape the indicies
        reshaped_indicies : torch.Tensor = rearrange(x, 'n (h w) -> n h w', h = 32, w = 32)

        # decode the tensor
        latent_tensor : torch.Tensor = self.quantizer.get_output_from_indices(reshaped_indicies)
        latent_tensor : torch.Tensor = rearrange(latent_tensor, 'n h w c -> n c h w')

        # forward pass the model
        decoded_tensor = self.decoder(latent_tensor)
        return decoded_tensor

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
    print("SIAK")

    m = LigweightAutoencoderRK512(1, 1)

    # ld = torch.load('./runs/cockatoo_jellyfish/weights/weights_best.pt', map_location = 'cpu')
    # m.load_state_dict(ld['autoencoder'])

    # m.freeze_layers_except(['encoder.output_projection', 'decoder.conv_project'])

    # for pname, pobj in m.named_parameters():
    #     print(pname, '\t' ,pobj.requires_grad)


    t = torch.rand(1, 1, 512, 512)
    flatten_z_quant, flatten_indicies, onehot_indicies = m.forward_encoder_tokens_targets(t)
    print(flatten_z_quant.shape)   # N 1024 8
    print(flatten_indicies.shape)  # N 1024
    print(onehot_indicies.shape)   # N 1024 256
    
    print('---')

    y = m.forward_decoder_tokens(flatten_indicies)
    print(y.shape)