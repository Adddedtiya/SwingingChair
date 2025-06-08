### Simple Autoencoder Wrapper Training Thing ###

import numpy as np

import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch.utils.data   import DataLoader
from .dictonary_tracker import GenericDictonaryTracker

from einops import rearrange, reduce, repeat, einsum

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

from ..models.model_siav import LigweightAutoencoder


class WrapperSAE:
    def __init__(
            self,
            input_channels  : int,
            output_channels : int,
            latent_size     : int,
            device          : torch.device = 'cpu' 
        ):

        # get the current device
        self.device = torch.device(device)
        self.latent_size = latent_size

        # create the autoencoder
        self.autoencoder = LigweightAutoencoder(
            input_channels  = input_channels,
            latent_size     = latent_size,
            output_channels = output_channels 
        ).to(self.device)

        # create the optimizer
        self.optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr = 2e-4)

        # create a static noise
        self.static_noise = torch.rand(1, self.latent_size, device = self.device)

    def save_state(self, fpath : str) -> None:
        x = {
            'static_noise'         : self.static_noise,
            'autoencoder'  : self.autoencoder.state_dict(),
            'optimizer'            : self.optimizer.state_dict()
        }
        # save the data dictonary
        torch.save(x, fpath)

    def load_state(self, fpath : str) -> None:
        state_dictonary = torch.load(
            fpath, map_location = self.device,
            weights_only = True
        )
        self.autoencoder.load_state_dict(state_dictonary['autoencoder'])
        self.optimizer.load_state_dict(state_dictonary['optimizer'])


    def __compute_metrics(self, x_real : torch.Tensor, y_pred : torch.Tensor) -> dict[str, float]:
        # compute the metrics
        ssim = structural_similarity_index_measure(y_pred, x_real).item()
        psnr = peak_signal_noise_ratio(y_pred, x_real).item()

        # cast the metric to float for sanity
        ssim = float(ssim)
        psnr = float(psnr)

        # package to dict
        xdict = {
            'ssim' : ssim,
            'psnr' : psnr
        }
        return xdict


    def train_single_batch(self, image_tensor : torch.Tensor) -> dict[str, float]:

        # grab the input_tensor size
        # batch_size, image_channels, image_H, image_W = image_tensor.shape

        # ensure the models are in Train mode
        self.autoencoder.train()

        # move the tensor to device
        image_tensor = image_tensor.to(self.device)

        # forward pass autoencoder
        reconstructed_image : torch.Tensor = self.autoencoder(image_tensor)

        # compute the loss and backprop
        recon_loss = F.mse_loss(reconstructed_image, image_tensor, reduction = 'mean')
        recon_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # compute the training metrics
        batch_stats = self.__compute_metrics(image_tensor, reconstructed_image)
        batch_stats['loss'] = float(recon_loss.item())

        return batch_stats

    def evaluate_single_batch(self, image_tensor : torch.Tensor) -> dict[str, float]:
        # ensure the models are in eval mode
        self.autoencoder.eval()

        # move the tensor to device
        image_tensor = image_tensor.to(self.device)

        # forward pass autoencoder
        reconstructed_image : torch.Tensor = self.autoencoder(image_tensor)

        # compute the evaluation metrics
        batch_stats = self.__compute_metrics(image_tensor, reconstructed_image)
        return batch_stats


    def train_single_epoch(self, training_dataloader : DataLoader) -> GenericDictonaryTracker:

        # parameters to track
        tracked_paramters = GenericDictonaryTracker()

        # Training Loop
        for sub_index, (image_tensor) in enumerate(training_dataloader):
            
            # display
            print("|- Index :", sub_index)

            # Train VAE
            print("|-- Training VAE")
            var_stats = self.train_single_batch(image_tensor)

            # track the stats
            tracked_paramters.append(var_stats)
        
        return tracked_paramters
    
    def evaluate_single_epoch(self, evaluation_dataloader : DataLoader) -> GenericDictonaryTracker:

        # parameters to track
        tracked_paramters = GenericDictonaryTracker()

        for sub_index, (image_tensor) in enumerate(evaluation_dataloader):
            
            # compute the batch metrics
            var_stats = self.evaluate_single_batch(image_tensor)

            # track the stats
            tracked_paramters.append(var_stats)

        return tracked_paramters
    
    def sample_generator(self, evaluation_dataloader : DataLoader) -> torch.Tensor:

        # ensure the models are in eval mode
        self.autoencoder.eval()
        for image_tensor in evaluation_dataloader:

            # move the tensor to device
            image_tensor = image_tensor.to(self.device)

            # forward pass autoencoder
            reconstructed_image : torch.Tensor = self.autoencoder(image_tensor)

            # we only want 1 guess
            break

        return reconstructed_image