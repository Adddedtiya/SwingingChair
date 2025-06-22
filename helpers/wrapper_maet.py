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
from torchmetrics.functional.classification import multiclass_accuracy

class WrapperMAET:
    def __init__(
            self,
            autoencoder     : nn.Module,
            model           : nn.Module,
            masked_ratio    : float        = 0.4,
            latent_size     : int          = 8,
            device          : torch.device = 'cpu' 
        ):

        # get the current device
        self.device = torch.device(device)
        self.latent_size = latent_size
        self.masked_ratio = masked_ratio

        # create the autoencoder and freeze the model autoencoder gradients
        self.autoencoder = autoencoder.to(self.device)
        self.autoencoder.freeze_layers_except([])
        self.autoencoder.eval()


        # create the model
        self.model = model.to(device)

        # create the optimizer
        self.optimizer = torch.optim.AdamW(self.model, lr = 2e-4)

        # create a static noise
        self.static_noise = torch.rand(1, self.latent_size, device = self.device)

    def save_state(self, fpath : str) -> None:
        x = {
            'static_noise' : self.static_noise,
            'model'        : self.model.state_dict(),
            'optimizer'    : self.optimizer.state_dict()
        }
        # save the data dictonary
        torch.save(x, fpath)

    def load_state(self, fpath : str) -> None:
        state_dictonary = torch.load(
            fpath, map_location = self.device,
            weights_only = True
        )
        self.model.load_state_dict(state_dictonary['model'])
        self.optimizer.load_state_dict(state_dictonary['optimizer'])


    def __compute_image_metrics(self, x_real : torch.Tensor, y_pred : torch.Tensor) -> dict[str, float]:
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

    def __compute_classification_metrics(self, x_real : torch.tensor, y_logits : torch.Tensor) -> dict[str, float]:

        # compute the accuracy
        top_1 = multiclass_accuracy(y_logits, x_real, num_classes = 256, top_k = 1).item()
        top_5 = multiclass_accuracy(y_logits, x_real, num_classes = 256, top_k = 5).item()

        return {
            'top_1' : float(top_1),
            'top_5' : float(top_5)
        }

    ### internal model wrappers just so i have type completions..
    
    def autoencoder_encoder_tokens_targets(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.autoencoder.forward_encoder_tokens_targets(x)

    def autoencoder_decoder_tokens(self, x : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.autoencoder.forward_decoder_tokens(x)

    def create_random_masked_indicies(self, batch_size : int = 1) -> torch.Tensor:
        return self.model.create_random_masked_indicies(self.masked_ratio, self.device, batch_size)

    def model_forward_masked_loss(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        return self.model.forward_masked_loss(x, y, self.masked_ratio)

    def model_forward_indicies_logits(self, x : torch.Tensor, masked_indicies : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_indicies(x, masked_indicies)

    ### Training Code here 

    def train_single_batch(self, image_tensor : torch.Tensor) -> dict[str, float]:

        # missing token prediction 

        # ensure the models are in Train mode
        self.model.train()

        # grab the input_tensor size
        batch_size, image_channels, image_H, image_W = image_tensor.shape
        
        # move the tensor to device
        image_tensor = image_tensor.to(self.device)

        # forward pass autoencoder
        flatten_z_quant, flatten_indicies, onehot_indicies = self.autoencoder_encoder_tokens_targets(image_tensor)

        # create random indicies
        # random_masked_indicies = self.create_random_masked_indicies(batch_size)

        # compute the loss
        cross_loss = self.model_forward_masked_loss(flatten_z_quant, onehot_indicies)
        cross_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        batch_stats = {
            'cross_entropy_loss' : cross_loss.item()
        }
        return batch_stats

    def evaluate_single_batch(self, image_tensor : torch.Tensor) -> dict[str, float]:
        # ensure the models are in eval mode
        self.model.eval()

        # move the tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # grab the input_tensor size
        batch_size, image_channels, image_H, image_W = image_tensor.shape

        # forward pass the encoder first
        flatten_z_quant, flatten_indicies, onehot_indicies = self.autoencoder_encoder_tokens_targets(image_tensor)

        # create random indicies
        random_masked_indicies = self.create_random_masked_indicies(batch_size)

        # forward the model with the randomly generated missing tokens 
        predicted_indicies, predicted_logits = self.model_forward_indicies_logits(flatten_z_quant, random_masked_indicies)

        # compute the output as classification task metric
        acc_stats = self.__compute_classification_metrics(predicted_logits, flatten_indicies)

        # grab the predicted indicies
        flatten_indicies[:, random_masked_indicies] = predicted_indicies[:, random_masked_indicies]

        # reconstruct the image based on the new indicies
        reconstructed_image = self.autoencoder_decoder_tokens(flatten_indicies)

        # compute the evaluation metrics
        img_stats = self.__compute_image_metrics(image_tensor, reconstructed_image)
        
        # merge the stats
        batch_stats = {**acc_stats, **img_stats}

        # compute the cross entropy as well !
        cross_loss = self.model_forward_masked_loss(flatten_z_quant, onehot_indicies)
        cross_loss = float(cross_loss.item())
        batch_stats['cross_entropy_loss'] = cross_loss

        return batch_stats


    def train_single_epoch(self, training_dataloader : DataLoader) -> GenericDictonaryTracker:

        # parameters to track
        tracked_paramters = GenericDictonaryTracker()

        # Training Loop
        for sub_index, (image_tensor) in enumerate(training_dataloader):
            
            # display
            print("|- Index :", sub_index)

            # Train VAE
            print("|-- Training MAET")
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