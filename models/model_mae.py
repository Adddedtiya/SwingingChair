import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from .model_sitv import *

# Implement MAE Models !

class SimpleMAE(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, image_size : int, patch_size : int, latent_size : int, encoder_depth : int, decoder_depth : int, heads : int, ff_dim : int):
        super().__init__()

        self.wrapped_encoder = BasicViTEncoder(
            input_channels = input_channels,
            image_size     = image_size,
            patch_size     = patch_size,
            latent_size    = latent_size,
            depth          = encoder_depth,
            heads          = heads,
            ff_dim         = ff_dim 
        )

        self.wrapped_decoder = BasicViTDecoder(
            output_channels = output_channels,
            image_size      = image_size,
            patch_size      = patch_size,
            latent_size     = latent_size,
            depth           = decoder_depth,
            heads           = heads,
            ff_dim          = ff_dim
        )

        # [MASK] token - for the decoder
        self.mask_token = nn.Parameter(torch.randn(self.wrapped_decoder.latent_size))

    def forward(self, x : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        
        ## Encoder Part of the Model ##

        # get the forward exec variable shape
        tensor_device = x.device
        batch_size, _, _, _ = x.shape

        # flattent the original image patches first + positional encoding
        flatten_input_patches = self.wrapped_encoder.flatten_to_patches(x)
        
        # select the patches from the decoder
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)

        # select patches form indicies
        selected_input_patches = flatten_input_patches[selected_batch_range, visible_indicies]

        # forward pass towards the encoder
        selected_patches_embedding = self.wrapped_encoder.transfomer(selected_input_patches)

        ## Decoder Part of the Model ##

        # create the decoder tokens + fill it with masks
        decoder_tokens = torch.zeros(batch_size, self.wrapped_encoder.total_patches, self.wrapped_encoder.latent_size, device = tensor_device)
        decoder_tokens[:, :] = self.mask_token

        # replace at location where the original images have patches
        decoder_tokens[selected_batch_range, visible_indicies] = selected_patches_embedding

        # crate the input embedding from input tensor with positional embedding (N, L, E)
        decoder_input_embbedding = decoder_tokens + self.wrapped_decoder.pos_embedding

        # pass the input embedding into the transfomer for decoding (N, L, E)
        decoded_embedding = self.wrapped_decoder.transfomer(decoder_input_embbedding)

        # project the image to original shape
        decoded_image = self.wrapped_decoder.decode_to_pixels(decoded_embedding)

        return decoded_image

    def create_random_visible_indicies(self, visible_patches : float = 0.5, device = 'cpu', batch_size : int = 1) -> torch.Tensor:
        # create the indicies
        path_ratio   = int(visible_patches * self.wrapped_encoder.total_patches)
        rand_indices = torch.rand(batch_size, self.wrapped_encoder.total_patches, device = device).argsort(dim = -1)
        
        # select the indicies
        visible_indicies = rand_indices[:, :path_ratio]
        return visible_indicies
    

    def reconstruct_visible_patches(self, input_image_tensor : torch.Tensor, visible_indicies : torch.Tensor) -> torch.Tensor:
        
        # flatten the input image
        flatten_input_patches =  rearrange(
            input_image_tensor, 
            "n c (h ph) (w pw) -> n (h w) (ph pw c)", 
            ph = self.wrapped_encoder.patch_height, pw = self.wrapped_encoder.patch_width
        )

        # torch information
        tensor_device = input_image_tensor.device
        batch_size, _, _, _ = input_image_tensor.shape

        # create batch indexes
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)
        
        # select the patches
        visible_patches = flatten_input_patches[selected_batch_range, visible_indicies]

        # reconstrcut tensor
        target_flatten_tensor = torch.zeros_like(flatten_input_patches, device = tensor_device)
        target_flatten_tensor[selected_batch_range, visible_indicies] = visible_patches

        image_tensor = rearrange(
            target_flatten_tensor, 
            "n (h w) (ph pw c) -> n c (h ph) (w pw)", 
            ph = self.wrapped_decoder.patch_height, 
            pw = self.wrapped_decoder.patch_width,
            c  = self.wrapped_decoder.output_channels,
            h  = int(self.wrapped_decoder.image_height // self.wrapped_decoder.patch_height),
            w  = int(self.wrapped_decoder.image_width  // self.wrapped_decoder.patch_width)
        )

        return image_tensor



if __name__ == "__main__":
    print(" MAE Models ")

    model = SimpleMAE(
        3, 3, 
        128, 16, 2048, 3, 3,8, 2048
    )
    t = torch.rand(1, 3, 128, 128)
    k = model.create_random_visible_indicies(0.3)
    y = model(t, k)
    print(y.shape)
    
