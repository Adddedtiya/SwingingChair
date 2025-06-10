import torch
import torch.nn as nn
from einops     import rearrange, reduce, repeat

class SimpleFeedForward(nn.Module):
    def __init__(self, dim : int, hidden_dim : int, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BasicAttention(nn.Module):
    def __init__(self, dim : int, heads : int = 8, dim_head : int = 64, dropout = 0.0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend  = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class CustomTransformer(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, dim_head : int, mlp_dim : int, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BasicAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                SimpleFeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class TransEncoder(nn.Module):
    def __init__(self, dim : int, depth : int, heads : int, mlp_dim : int, dropout = 0.0):
        super().__init__()
        self.transformer_encoder = CustomTransformer(
            dim, depth, heads, heads, mlp_dim, dropout
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x) 





######################################
# Integrated Machine Learning Models #
######################################

class BasicViTEncoder(nn.Module):
    def __init__(self, input_channels : int, image_size : int, patch_size : int, latent_size : int, depth : int, heads : int, ff_dim : int):
        super().__init__()
        
        # setup and sanity check
        self.image_height, self.image_width = (image_size, image_size)
        self.patch_height, self.patch_width = (patch_size, patch_size)
        assert (self.image_height % self.patch_height == 0) and (self.image_width % self.patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # calculate the total patches, and the flatten patch size
        self.flatten_patch_size = input_channels * self.patch_height * self.patch_width
        self.total_patches      = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, latent_size))

        self.project_embbedding = nn.Sequential(
            nn.LayerNorm(self.flatten_patch_size),
            nn.Linear(self.flatten_patch_size, latent_size),
            nn.LayerNorm(latent_size)
        ) 

        # the transfomer model it self 
        self.transfomer = TransEncoder(
            dim     = latent_size,
            depth   = depth,
            heads   = heads,
            mlp_dim = ff_dim,
        )

    def flatten_to_patches(self, x : torch.Tensor) -> torch.Tensor:

        # reshape and flatten the tensor (N, C, H, W) -> (N, patch_count, flatten_patch)
        x = rearrange(x, "n c (h ph) (w pw) -> n (h w) (ph pw c)", ph = self.patch_height, pw = self.patch_width)

        # project the flatten image to the shape (N, patch_count, latent_size)
        x = self.project_embbedding(x)
        
        # add positional information on the embedding
        x = x + self.pos_embedding

        return x


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # convert the image tensor to projected flatten patches with positional encoding
        flatten_patches = self.flatten_to_patches(x)

        # pass the encoded patches to the transfomer (N, L, E)
        encoded_tensor = self.transfomer(flatten_patches)

        return encoded_tensor


class BasicViTDecoder(nn.Module):
    def __init__(self, output_channels : int, image_size : int, patch_size : int, latent_size : int, depth : int, heads : int, ff_dim : int):
        super().__init__()

        # setup and sanity check
        self.image_height, self.image_width = (image_size, image_size)
        self.patch_height, self.patch_width = (patch_size, patch_size)
        assert (self.image_height % self.patch_height == 0) and (self.image_width % self.patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # calculate the total patches, and the flatten patch size
        self.flatten_patch_size = output_channels * self.patch_height * self.patch_width
        self.total_patches      = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        # remeber the output channels and size
        self.output_channels = output_channels
        self.latent_size     = latent_size

        # learnable ? positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, latent_size))

        self.transfomer = TransEncoder(
            dim     = latent_size,
            depth   = depth,
            heads   = heads,
            mlp_dim = ff_dim,
        )

        # linearly project from embbedding to pixels
        self.project_patches = nn.Sequential(
            nn.Linear(latent_size, self.flatten_patch_size, bias = True),
            nn.Sigmoid()
        )
    
    def decode_to_pixels(self, embedding : torch.Tensor) -> torch.Tensor:

        # convert the embeeding to pixels patches (N, L, E) -> (N, L, P) 
        flatten_patches = self.project_patches(embedding)

        # re-arrange the tensor to PyTorch Image Tensor (N, C, H, W)
        image_tensor = rearrange(
            flatten_patches, 
            "n (h w) (ph pw c) -> n c (h ph) (w pw)", 
            ph = self.patch_height, 
            pw = self.patch_width,
            c  = self.output_channels,
            h  = int(self.image_height // self.patch_height),
            w  = int(self.image_width  // self.patch_width)
        )

        return image_tensor


    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # crate the input embedding from input tensor with positional embedding (N, L, E)
        input_embbedding = x + self.pos_embedding

        # pass the input embedding into the transfomer for decoding (N, L, E)
        decoded_embedding = self.transfomer(input_embbedding)

        image_tensor = self.decode_to_pixels(decoded_embedding)
        return image_tensor


class SimpleViSaE(nn.Module):
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
        # Latent Mean (for now....)
        # self.mask_token = nn.Parameter(torch.randn(self.wrapped_decoder.latent_size))

    def forward_encoder(self, x : torch.Tensor) -> torch.Tensor:
        
        # flattent the original image patches first + positional encoding
        flatten_input_patches = self.wrapped_encoder.flatten_to_patches(x)

        # pass the model input the transfomer (N L E -> N L E)
        image_patched_embedding = self.wrapped_encoder.transfomer(flatten_input_patches)

        # compute reduction of the mean over the (N E)
        latent_image_embedding = reduce(image_patched_embedding, "n l e -> n e", reduction = 'mean')

        return latent_image_embedding

    def forward_decoder(self, x : torch.Tensor) -> torch.Tensor:

        # create the decoder tokens by repeating the original input
        decoder_tokens = repeat(x, "n e -> n l e", l = self.wrapped_decoder.total_patches)

        # crate the input embedding from input tensor with positional embedding (N, L, E)
        input_embbedding = decoder_tokens + self.wrapped_decoder.pos_embedding

        # pass the input embedding into the transfomer for decoding (N, L, E)
        decoded_embedding = self.wrapped_decoder.transfomer(input_embbedding)

        # project the image to original shape
        image_tensor = self.wrapped_decoder.decode_to_pixels(decoded_embedding)
        return image_tensor

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # forward the model to create a latent shape
        latent_z = self.forward_encoder(x)

        # forward to reconnstruct the image
        reco_img = self.forward_decoder(latent_z)

        return reco_img



if __name__ == "__main__":
    print("Model Transfomer Thing")


    model = SimpleViSaE(
        3, 3, 128, 16, 512, 12, 12, 7, 728
    )

    t = torch.rand(1, 3, 128, 128)
    y = model(t)
    print(y.shape)

    from torchinfo import summary
    summary(model, input_data = t)
