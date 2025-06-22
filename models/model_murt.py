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


class SimpleMurq(nn.Module):
    def __init__(self, input_size : int, length : int, output_classes : int, latent_size : int, depth : int, heads : int, ff_dim : int):
        super().__init__()

        self.input_size      = input_size
        self.latent_size     = latent_size
        self.sequence_length = length
        self.output_classes  = output_classes

        # learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.sequence_length, self.latent_size))

        # [MASK] token
        self.mask_token = nn.Parameter(torch.randn(self.latent_size), requires_grad = False)

        # project to model input
        self.project_input = nn.Linear(self.input_size, self.latent_size)
        
        # the transfomer model it self 
        self.transfomer = TransEncoder(
            dim     = latent_size,
            depth   = depth,
            heads   = heads,
            mlp_dim = ff_dim,
        )

        # project to model output
        self.project_output = nn.Linear(self.latent_size, self.output_classes)


    def forward(self, x : torch.Tensor, masked_indicies : torch.Tensor) -> torch.Tensor:

        # input tensor metadata
        batch_size, _, _ = x.shape
        tensor_device    = x.device
        
        # project the input to shape (N, L, C) -> (N, L, E)
        projected_input : torch.Tensor = self.project_input(x)

        # select the patches from the decoder
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)
        
        # swap some of the projected tokens for [MASK] token
        projected_input[selected_batch_range, masked_indicies] = self.mask_token

        # add positional tokens
        projected_input = projected_input + self.pos_embedding

        # input with masks 
        decoder_output = self.transfomer(projected_input)

        # project the output (N, L, E)
        projected_output = self.project_output(decoder_output)

        # project the output
        return projected_output

    def create_random_masked_indicies(self, masked_patches : float = 0.4, device = 'cpu', batch_size : int = 1) -> torch.Tensor:
        # create the indicies
        path_ratio   = int(masked_patches * self.sequence_length)
        rand_indices = torch.rand(batch_size, self.sequence_length, device = device).argsort(dim = -1)
        
        # select the indicies
        masked_indicies = rand_indices[:, :path_ratio]
        return masked_indicies

    def forward_masked_loss(self, x : torch.Tensor, y : torch.Tensor, masked_ratio : float = 0.4) -> torch.Tensor:

        # input tensor metadata
        batch_size, _, _ = x.shape
        tensor_device    = x.device

        # create the random masked tensor
        mask_indicies_tensor = self.create_random_masked_indicies(masked_ratio, tensor_device, batch_size)
        #print(mask_indicies_tensor)
        
        # select the patches from the decoder
        selected_batch_range = torch.arange(batch_size, device = tensor_device).reshape(batch_size, 1)

        # forward pass the model
        predicted_logits = self.forward(x, mask_indicies_tensor)
        
        # select the tokens that was [MASK] (N, L, E)
        masked_patches = predicted_logits[selected_batch_range, mask_indicies_tensor]
        #print(masked_patches.shape)

        # okay now select the target patches (y is supposed to be one-hot tensor) (N, L, H)
        selected_y = y[selected_batch_range, mask_indicies_tensor]
        #print(selected_y.shape)

        # batch flatten prediction
        flatten_logits = rearrange(masked_patches, 'n l e -> (n l) e')
        # print(flatten_logits.shape)
        # print(flatten_logits)

        # batch flatten targets
        flatten_hots = rearrange(selected_y, 'n l h -> (n l) h')
        # print(flatten_hots.shape)
        # print(flatten_hots)

        # compute the loss
        cross_entropy = nn.functional.cross_entropy(flatten_logits, flatten_hots)
        return cross_entropy

    def forward_indicies(self, x : torch.Tensor, masked_indicies : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        # forward pass (N L E) and get one hot indicies (N L H)
        predicted_logits = self.forward(x, masked_indicies)
        predicted_logits = nn.functional.softmax(predicted_logits, dim = -1)

        # convert the predicted logits to one hot indicies
        predicted_indicies = torch.argmax(predicted_logits, dim = -1)
        return predicted_indicies, predicted_logits


if __name__ == "__main__":
    print("Bert Like Model")

    model = SimpleMurq(
        input_size     = 8,
        length         = 1024,
        output_classes = 256,
        latent_size    = 768,
        depth          = 13,
        heads          = 12,
        ff_dim         = 1024
    )
    b = 2

    m = model.create_random_masked_indicies(0.1, batch_size = b)
    x = torch.rand(b, 1024, 8, requires_grad = True)
    y = torch.rand(b, 1024, 256).softmax(dim = -1)

    # l = model.forward_masked_loss(x, y, 0.1)
    # print(l)
    
    # i, loits = model.forward_indicies(x, m)
    # # print(i.shape)
    # # print(i[:, m].shape)
    # # print("<<", m)
    # # i[:, m] = 1000
    # # print(i)

    # print(loits.shape)
    from torchinfo import summary
    summary(model, input_data = (x, m)) # 1.2 GB per batch








































