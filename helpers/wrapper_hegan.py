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

class WrapperHEGAN:
    def __init__(
            self,
            autoencoder     : nn.Module,
            discriminator   : nn.Module,
            latent_size     : int          = 8,
            device          : torch.device = 'cpu' 
        ):

        # get the current device
        self.device = torch.device(device)
        self.latent_size = latent_size

        # create the autoencoder
        self.autoencoder   = autoencoder.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # create the optimizer
        self.optimizer_ae = torch.optim.AdamW(self.autoencoder.parameters(),   lr = 2e-4)
        self.optimizer_dc = torch.optim.AdamW(self.discriminator.parameters(), lr = 2e-4)

        # create a static noise
        self.static_noise = torch.rand(1, self.latent_size, device = self.device)

    def save_state(self, fpath : str) -> None:
        x = {
            'static_noise'  : self.static_noise,
            'autoencoder'   : self.autoencoder.state_dict(),
            'discriminator' : self.discriminator.state_dict(),
            'optim_ae'      : self.optimizer_ae.state_dict(),
            'optim_dc'      : self.optimizer_dc.state_dict(),
        }
        # save the data dictonary
        torch.save(x, fpath)

    def load_state(self, fpath : str) -> None:
        state_dictonary = torch.load(
            fpath, map_location = self.device,
            weights_only = True
        )
        self.autoencoder.load_state_dict(state_dictonary['autoencoder'])
        self.discriminator.load_state_dict(state_dictonary['discriminator'])
        self.optimizer_ae.load_state_dict(state_dictonary['optim_ae'])
        self.optimizer_dc.load_state_dict(state_dictonary['optim_dc'])


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

    def ae_forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)
    
    def dc_forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.discriminator.forward_discriminator(x)

    def train_single_batch(self, image_tensor : torch.Tensor) -> dict[str, float]:

        # grab the input_tensor size
        batch_size, image_channels, image_H, image_W = image_tensor.shape

        # create targets
        target_real = torch.ones(batch_size,  1, 16, 16, device = self.device)
        target_fake = torch.zeros(batch_size, 1, 16, 16, device = self.device)

        # ensure the models are in Train mode
        self.autoencoder.train()

        # move the tensor to device
        image_tensor = image_tensor.to(self.device)

        # forward pass to ae
        reconstructed_image = self.ae_forward(image_tensor)

        # forward discriminator
        pred_real = self.dc_forward(image_tensor)
        pred_fake = self.dc_forward(reconstructed_image)

        # Compute the loss functions
        loss_real = F.binary_cross_entropy_with_logits(pred_real, target_real)
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, target_fake)
        loss_advs = loss_real + loss_fake

        # compute the distance loss
        loss_smooth = F.smooth_l1_loss(reconstructed_image, image_tensor)

        # combine the loss
        combined_loss = (loss_smooth + loss_advs) / 2
        combined_loss.backward()
        
        self.optimizer_ae.step()
        self.optimizer_dc.step()

        self.optimizer_dc.zero_grad()
        self.optimizer_ae.zero_grad()

        # compute the training metrics
        batch_stats = self.__compute_metrics(image_tensor, reconstructed_image)
        batch_stats['loss']        = float(combined_loss.item())
        batch_stats['loss_real']   = float(loss_real.item())
        batch_stats['loss_fake']   = float(loss_fake.item())
        batch_stats['loss_smooth'] = float(loss_smooth.item())

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
            print("|-- Training DISC")
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

        # interleave images
        interleaved_tensor = torch.cat([image_tensor, reconstructed_image], dim = -1)

        return interleaved_tensor