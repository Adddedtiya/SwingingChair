import torch
import torch.nn as nn
from einops     import rearrange, reduce, repeat
from vector_quantize_pytorch import VectorQuantize


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

class SimpleTransfomer(nn.Module):
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

######################################
# Integrated Machine Learning Models #
######################################

class BasicViTEncoder(nn.Module):
    def __init__(self, input_channels : int, image_size : int, patch_size : int, t_latent_size : int, t_depth : int, t_heads : int, t_dim : int, embedding_size : int):
        super().__init__()
        
        # setup and sanity check
        self.image_height, self.image_width = (image_size, image_size)
        self.patch_height, self.patch_width = (patch_size, patch_size)
        assert (self.image_height % self.patch_height == 0) and (self.image_width % self.patch_width) == 0, 'Image dimensions must be divisible by the patch size.'

        # calculate the total patches, and the flatten patch size
        self.flatten_patch_size = input_channels * self.patch_height * self.patch_width
        self.total_patches      = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        # setup the latent size too
        self.latent_size = t_latent_size

        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, self.latent_size), requires_grad = False)

        # project the input
        self.project_input = nn.Linear(self.flatten_patch_size, self.latent_size)

        # the transfomer model it self 
        self.transfomer = SimpleTransfomer(
            dim      = self.latent_size,
            depth    = t_depth,
            heads    = t_heads,
            dim_head = t_heads,
            mlp_dim  = t_dim
        )

        # project the output
        self.project_output = nn.Linear(self.latent_size, embedding_size)

    def flatten_to_patches(self, x : torch.Tensor) -> torch.Tensor:

        # reshape and flatten the tensor (N, C, H, W) -> (N, patch_count, flatten_patch)
        x = rearrange(x, "n c (h ph) (w pw) -> n (h w) (ph pw c)", ph = self.patch_height, pw = self.patch_width)

        # project the flatten image to the shape (N, patch_count, latent_size)
        x = self.project_input(x)
        
        # add positional information on the embedding
        x = x + self.pos_embedding

        return x

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        # convert the image tensor to projected flatten patches with positional encoding
        flatten_patches = self.flatten_to_patches(x)

        # pass the encoded patches to the transfomer (N, L, E)
        encoded_tensor = self.transfomer(flatten_patches)

        # project it to the latent size 
        encoded_tensor = self.project_output(encoded_tensor)

        return encoded_tensor


class BasicViTDecoder(nn.Module):
    def __init__(self, output_channels : int, image_size : int, patch_size : int, t_latent_size : int, t_depth : int, t_heads : int, t_dim : int, embedding_size : int):
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
        self.latent_size     = t_latent_size

        # learnable ? positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_patches, self.latent_size), requires_grad = False)

        self.project_input = nn.Linear(embedding_size, self.latent_size)

        self.transfomer = SimpleTransfomer(
            dim      = self.latent_size,
            depth    = t_depth,
            heads    = t_heads,
            dim_head = t_heads,
            mlp_dim  = t_dim
        )

        # linearly project from embbedding to pixels
        self.project_patches = nn.Linear(self.latent_size, self.flatten_patch_size, bias = True)


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
        
        # project the embedding to latent size (N, L, Z) -> (N, L, E)
        x = self.project_input(x)

        # crate the input embedding from input tensor with positional embedding (N, L, E)
        input_embbedding = x + self.pos_embedding

        # pass the input embedding into the transfomer for decoding (N, L, E)
        decoded_embedding = self.transfomer(input_embbedding)

        image_tensor = self.decode_to_pixels(decoded_embedding)
        return image_tensor


class TransAutoencoderV512(nn.Module):
    def __init__(self, input_channels : int, output_channels : int, image_size : int):
        super().__init__()

        # setup size
        self.embedding_size = 8
        self.patch_size     = 16

        # transfomer model
        self.latent_size   = 768
        self.heads         = 13
        self.depth         = 15
        self.feed_forward  = 2048

        # vector quantize
        self.vq_dim  = 8
        self.vq_code = 8
        self.vq_size = 256

        self.encoder = BasicViTEncoder(
            input_channels = input_channels,
            image_size     = image_size,
            patch_size     = self.patch_size,
            t_latent_size  = self.latent_size,
            t_depth        = self.depth,
            t_heads        = self.heads,
            t_dim          = self.feed_forward,
            embedding_size = self.embedding_size
        )

        self.decoder = BasicViTDecoder(
            output_channels = output_channels,
            image_size      = image_size,
            patch_size      = self.patch_size,
            t_latent_size   = self.latent_size,
            t_depth         = self.depth,
            t_heads         = self.heads,
            t_dim           = self.feed_forward,
            embedding_size  = self.embedding_size
        )

        self.quantizer = VectorQuantize(
            dim               = self.vq_dim,
            codebook_size     = self.vq_size,
            codebook_dim      = self.vq_code,
            use_cosine_sim    = True,
            accept_image_fmap = False
        )

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

    def forward_encoder_tokens_targets(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        compressed_latent = self.encoder(x)
        z_q, i_c, _ = self.quantizer(compressed_latent)

        # flatten indicies tensors
        flatten_indicies : torch.Tensor = i_c.to(torch.long)
        
        # convert to one hot indicies
        onehot_indicies : torch.Tensor = nn.functional.one_hot(flatten_indicies, num_classes = self.vq_size)
        onehot_indicies : torch.Tensor = onehot_indicies.to(torch.float32)

        # flatten z space
        flatten_z_quant : torch.Tensor = z_q.detach()

        return (flatten_z_quant, flatten_indicies, onehot_indicies)

    def forward_decoder_tokens(self, x : torch.Tensor) -> torch.Tensor:

        # decode the tensor
        latent_tensor : torch.Tensor = self.quantizer.get_output_from_indices(x)

        # forward pass the model
        decoded_tensor = self.decoder(latent_tensor)
        return decoded_tensor

if __name__ == "__main__":
    print("TAME Like Model")

    model_tame = TransAutoencoderV512(1, 1, 512)
    x = torch.rand(1, 1, 512, 512)

    from torchinfo import summary
    summary(model_tame, input_data = x)