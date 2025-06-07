### Beta Variable AutoEncoder GAN ###

import numpy as np

import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch.utils.data   import DataLoader
from .dictonary_tracker import GenericDictonaryTracker

from einops import rearrange, reduce, repeat, einsum

from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure

class WrapperBVG:
    def __init__(
            self, 
            encoder : nn.Module,
            decoder : nn.Module,
            critic  : nn.Module,
            latent_size : int,
            device  : torch.device = 'cpu' 
        ):
        
        # Get the Device
        self.device = device
        self.latent_size = latent_size

        # Get the Encoder Optimzer
        self.encoder_model_vae = encoder.to(self.device)
        self.vae_optim_encoder = torch.optim.AdamW(self.encoder_model_vae.parameters(), lr = 2e-4)

        # Decoder - Generator with dual optimizer -> VAE and GAN domain
        self.decoder_generator_model = decoder.to(self.device)
        self.vae_optim_decoder   = torch.optim.AdamW(self.decoder_generator_model.parameters(), lr = 2e-4)
        self.gan_optim_generator = torch.optim.AdamW(self.decoder_generator_model.parameters(), lr = 2e-4)

        # Crtic -> GAN
        self.critic_gan_model = critic.to(self.device)
        self.gan_optim_critic = torch.optim.AdamW(self.critic_gan_model.parameters(), lr = 2e-4)

        self.static_noise = torch.rand(1, self.latent_size, device = self.device)

    def save_state(self, fpath : str) -> None:
        x = {
            'static_noise'              : self.static_noise,
            'vae_encoder_weights'       : self.encoder_model_vae.state_dict(),
            'vae_encoder_optim'         : self.vae_optim_encoder.state_dict(),
            'decoder_generator_weights' : self.decoder_generator_model.state_dict(),
            'vae_decoder_optim'         : self.vae_optim_decoder.state_dict(),
            'gan_generator_optim'       : self.gan_optim_generator.state_dict(),
            'gan_critic_weights'        : self.critic_gan_model.state_dict(),
            'gan_critic_optim'          : self.gan_optim_critic.state_dict()
        }
        # save the data dictonary
        torch.save(x, fpath)

    def __vae_reparameterize(self, mu : torch.Tensor, logvar : torch.Tensor, deterministic : bool = False) -> torch.Tensor:
        
        # if deterministic so no noise
        if deterministic:
            return mu
        
        # forward reparam, with noise added
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def __gradient_penalty(self, real : torch.Tensor, fake : torch.Tensor) -> torch.Tensor:
        batch_size, image_channels, image_H, image_W = real.shape

        # create alpha-noise and repeat
        alpha = torch.rand((batch_size))
        alpha = repeat(alpha, "n -> n c h w", c = image_channels, h = image_H, w = image_W)
        alpha = alpha.to(self.device)

        # interpolate between the real and fake image
        interpolate_real = real * alpha
        interpolate_fake = fake * (1 - alpha)
        interpolated_images = interpolate_real + interpolate_fake
        interpolated_images.requires_grad_(True)

        # compute the mixed score of critic
        mixed_scores = self.critic_gan_model(interpolated_images)

        # compute the gradient
        gradient = torch.autograd.grad(
            inputs        = interpolated_images,
            outputs       = mixed_scores,
            grad_outputs  = torch.ones_like(mixed_scores),
            create_graph  = True,
            retain_graph  = True,
        )[0]

        # project the gradient
        gradient = gradient.view(batch_size, -1)
        
        # compute the gradient norm
        gradient_norm = gradient.norm(2, dim = 1)
        gradient_norm = gradient_norm - 1
        gradient_norm = gradient_norm ** 2
        
        # get the mean as penalty
        gradient_penalty = torch.mean(gradient_norm)
        return gradient_penalty
    

    def train_vae_single_batch(self, image_tensor : torch.Tensor, beta_value : float = 0.8) -> dict[str, float]:

        # grab the input_tensor size
        batch_size, image_channels, image_H, image_W = image_tensor.shape

        # ensure the models are in Train mode
        self.encoder_model_vae.train()
        self.decoder_generator_model.train()

        # Ensure the model is clean
        self.encoder_model_vae.zero_grad()
        self.decoder_generator_model.zero_grad()

        # Ensure that optimzers are clean
        self.vae_optim_encoder.zero_grad()
        self.vae_optim_decoder.zero_grad()

        # move the tensor to device
        image_tensor = image_tensor.to(self.device)

        # forward pass from Encoder
        latent_mu, latent_log_var = self.encoder_model_vae(image_tensor)
        latent_mu      : torch.Tensor = latent_mu
        latent_log_var : torch.Tensor = latent_log_var
        latent_tensor = self.__vae_reparameterize(latent_mu, latent_log_var)

        # forward pass to Decoder
        reconstructed_tensor = self.decoder_generator_model(latent_tensor)

        # compute MSE loss
        loss_mse = F.mse_loss(reconstructed_tensor, image_tensor, reduction = 'mean')
        loss_mse = loss_mse / batch_size

        # compute KL-Divergence '0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)' Reparaprase
        loss_kld = -0.5 * torch.sum(1 + latent_log_var - latent_mu.pow(2) - latent_log_var.exp())
        loss_kld = loss_kld / batch_size

        # combine the loss + backpropagate
        loss_combined = loss_mse + (loss_kld * beta_value)
        loss_combined.backward()

        # update the model weights
        self.vae_optim_encoder.step()
        self.vae_optim_decoder.step()

        # Ensure that optimzers are clean
        self.vae_optim_encoder.zero_grad()
        self.vae_optim_decoder.zero_grad()

        # send the report back out
        batch_report = {
            'loss_mse'      : float(loss_mse.item()),
            'loss_kld'      : float(loss_kld.item()),
            'loss_combined' : float(loss_combined.item())
        }
        return batch_report

    def train_gan_critic_iter(self, dataloader : DataLoader, critic_iter : int = 1, lambda_gp : int = 10) -> dict[str, float]:

        # ensure the models are in right mode
        self.critic_gan_model.train()
        self.decoder_generator_model.eval()

        # Ensure that optimzers are clean
        self.gan_optim_critic.zero_grad()
        self.gan_optim_generator.zero_grad()

        # internal tracking for the critic loss
        critic_loss_list : list[float] = []

        # Training Loop For Critic
        for index, (image_tensor) in enumerate(dataloader):
            
            # clear grads in critic
            self.critic_gan_model.zero_grad()

            # grab the input_tensor size
            image_tensor : torch.Tensor = image_tensor.to(self.device)
            batch_size, image_channels, image_H, image_W = image_tensor.shape

            # create fake noise with no gradients
            with torch.no_grad():
                fake_noise = torch.rand((batch_size, self.latent_size)).to(self.device)
                fake_image = self.decoder_generator_model(fake_noise)
            
            # compute the critic results
            critic_real = self.critic_gan_model(image_tensor)
            critic_fake = self.critic_gan_model(fake_image)

            # compute the gradient penalty for the critic
            gradient_penalty = self.__gradient_penalty(image_tensor, fake_image)

            # compute the critic loss
            loss_critic = (torch.mean(critic_fake) - torch.mean(critic_real)) + (lambda_gp * gradient_penalty)
            loss_critic.backward()

            # step the and clear the critic
            self.gan_optim_critic.step()
            self.gan_optim_critic.zero_grad()

            # append the loss
            critic_loss_list.append(float(loss_critic.item()))

            # check if the current iter is above the select amount
            if (index + 1) >= critic_iter:
                break
        
        # free from the training loop

        # compute the average critic loss
        average_critic_loss = float(np.mean(critic_loss_list))

        # create a report and retun
        batch_report = {
            'critic_loss' : average_critic_loss
        }
        return batch_report
    
    def train_gan_generator_iter(self, image_tensor : torch.Tensor, count : int = 1) -> dict[str, float]:
        
        # grab the input_tensor size
        batch_size, image_channels, image_H, image_W = image_tensor.shape

        # ensure the models are in right mode
        self.critic_gan_model.eval()
        self.decoder_generator_model.train()

        # Ensure that optimzers are clean
        self.gan_optim_critic.zero_grad()
        self.gan_optim_generator.zero_grad()

        # internal tracking for the critic loss
        generator_loss_list : list[float] = []
        
        # loop 
        for _ in range(count):
            
            # ensure no gradients in generator
            self.decoder_generator_model.zero_grad()

            # randomly sample from latent
            fake_noise = torch.rand((batch_size, self.latent_size)).to(self.device)
            fake_image = self.decoder_generator_model(fake_noise)

            # compute the critic result
            critic_mesurement = self.critic_gan_model(fake_image)
            
            # compute the loss
            loss_gen = -torch.mean(critic_mesurement)
            loss_gen.backward()

            # update the weights
            self.gan_optim_generator.step()

            # append the loss
            generator_loss_list.append(float(loss_gen.item()))

        # free from loop

        # compute the average generator loss
        average_generator_loss = float(np.mean(generator_loss_list))

        # create a report and retun
        batch_report = {
            'generator_loss' : average_generator_loss
        }
        return batch_report

    def sample_generator(self, amount : int = None) -> torch.Tensor:

        # set the decoder to eval
        self.decoder_generator_model.eval()

        # check if the amont is a number
        if amount:
            z = torch.rand(amount, self.latent_size).to(self.device)
        else:
            z = self.static_noise

        # forward pass
        with torch.no_grad():
            generated = self.decoder_generator_model(z)

        # return the generated tensor
        return generated

    def direct_reconstruction(self, image_tensor : torch.Tensor) -> torch.Tensor:

        # set the decoder-decoder to eval
        self.encoder_model_vae.eval()
        self.decoder_generator_model.eval()

        # move the device 
        image_tensor = image_tensor.to(self.device)

        # forward pass from Encoder
        latent_mu, latent_log_var = self.encoder_model_vae(image_tensor)
        latent_mu      : torch.Tensor = latent_mu
        latent_log_var : torch.Tensor = latent_log_var
        latent_tensor = self.__vae_reparameterize(latent_mu, latent_log_var, deterministic = True)

        # forward pass to Decoder
        reconstructed_tensor = self.decoder_generator_model(latent_tensor)

        # return the reconstructed image
        return reconstructed_tensor

    def train_single_epoch(
            self, 
            training_dataloader : DataLoader, 
            critic_dataloader   : DataLoader, 
            critic_iter         : int = 1, 
            critic_lgp          : int = 10, 
            vae_beta_value      : float = 0.8
        ) -> GenericDictonaryTracker:

        tracked_paramters = GenericDictonaryTracker()

        # Training Loop
        for sub_index, (image_tensor) in enumerate(training_dataloader):
            
            # display
            print("|- Index :", sub_index)

            # Train VAE
            print("|-- Training VAE")
            var_stats  = self.train_vae_single_batch(image_tensor, vae_beta_value)            

            # Train Critic
            print("|-- Training Critic")
            crit_stats = self.train_gan_critic_iter(critic_dataloader, critic_iter, critic_lgp)

            # Train Generator (VAE - Decoder)
            print("|-- Training Generator-Decoder")
            gen_stats = self.train_gan_generator_iter(image_tensor)

            # combine all stats
            batch_stats = {**var_stats, **crit_stats, **gen_stats}
            tracked_paramters.append(batch_stats)
        
        return tracked_paramters

    def evaluate_single_epoch(
            self,
            evaluation_dataloader : DataLoader
        ) -> GenericDictonaryTracker:

        # parameters to track
        tracked_paramters = GenericDictonaryTracker()

        for real_image_tensor in evaluation_dataloader:
            
            with torch.no_grad():
                reconstruction_tensor = self.direct_reconstruction(real_image_tensor)
            
            # compute the metrics
            ssim = structural_similarity_index_measure(reconstruction_tensor, real_image_tensor).item()
            psnr = peak_signal_noise_ratio(reconstruction_tensor, real_image_tensor).item()

            # cast to floats
            ssim = float(ssim)
            psnr = float(psnr)

            # convert to dict and append
            tracked_paramters.append({
                'ssim' : ssim,
                'psnr' : psnr
            })
        
        return tracked_paramters