import os
import argparse

import torch
import random
import numpy as np

from data.dataset_reconstruction import ReconstructionDataset
from helpers.dictonary_tracker   import TrackerAndLogger
from helpers.wrapper_bvg         import WrapperBVG
from models.model_sibav          import SimpleCritic, SimpleEncoder, SimpleDecoder

# Deterministic Algorithms
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("## Training VAE-WGAN ##")

    # setup args
    parser = argparse.ArgumentParser(description = "VAE-GAN Training configuration")
    parser.add_argument('--total_epochs', type = int, default = 512)
    parser.add_argument('--batch_size',   type = int, default = 16)
    parser.add_argument('--load_threads', type = int, default = 4)
    parser.add_argument('--latent_size',  type = int, default = 256)
    parser.add_argument('--dataset_root', type = str, default = '')
    parser.add_argument('--name'        , type = str, default = '')
    parser.add_argument('--memory_cache', action = 'store_true')

    # Prase the Arguemnts
    parsed_args  = parser.parse_args()    
    total_epochs : int  = parsed_args.total_epochs
    batch_size   : int  = parsed_args.batch_size
    load_threads : int  = parsed_args.load_threads
    latent_size  : int  = parsed_args.latent_size
    dataset_root : str  = parsed_args.dataset_root
    exp_name     : str  = parsed_args.name
    memory_cache : bool = parsed_args.memory_cache

    print("| Pytorch Model Training !")
    print("| Total Epoch :", total_epochs)
    print("| Batch Size  :", batch_size)
    print("| Workers     :", load_threads)
    print("| Device      :", device)
    print("| Latent      :", latent_size)
    print("| Name        :", exp_name)
    print("| Mem-Only    :", memory_cache)

    # create Trackers
    logger = TrackerAndLogger('./runs', exp_name, metric_to_track = 'ssim')

    # Create The Model
    model_encoder = SimpleEncoder(1, latent_size)
    model_decoder = SimpleDecoder(latent_size, 1)
    model_critic  = SimpleCritic(1)

    # Training Helper Wrapper
    model_wrapper = WrapperBVG(
        encoder = model_encoder,
        decoder = model_decoder,
        critic  = model_critic,
        device  = device,
        latent_size = latent_size
    )

    # Create Dataset
    training_vae_dataset = ReconstructionDataset(dataset_root, 'train', memory_cache)
    training_crt_dataset = ReconstructionDataset(dataset_root, 'train', memory_cache)
    evaluation_dataset   = ReconstructionDataset(dataset_root, 'eval',  memory_cache)
    #testing_dataset   = ReconstructionDataset(dataset_root, 'test',  memory_cache)

    # create dataloaders - main loop + critic
    dataloader_train_main = training_vae_dataset.create_dataloader(batch_size, load_threads, device)
    dataloader_train_crit = training_crt_dataset.create_dataloader(batch_size, load_threads, device)

    # create dataloader for test and eval
    #dataloader_test = testing_dataset.create_dataloader(1, 0, device)
    dataloader_eval = evaluation_dataset.create_dataloader(batch_size, load_threads, device)

    print("| Setup Complete Start Training !")    
    for current_epoch in range(total_epochs):

        print(f"| Current Epoch {current_epoch + 1}/{total_epochs}")
        
        # Train the Model For a single epoch
        train_stats = model_wrapper.train_single_epoch(
            dataloader_train_main,
            dataloader_train_crit,
            critic_iter    = 5,
            critic_lgp     = 10,
            vae_beta_value = 0.8
        )

        # Evaluate the model
        print("| Training Complete, Evaluating...")
        eval_stats = model_wrapper.evaluate_single_epoch(dataloader_eval)

        # track the stats
        logger.append_epoch(
            train = train_stats,
            eval  = eval_stats
        )

        # check if the current epoch is best
        if logger.current_is_best():
            
            print("| Current Epoch is Best !")

            # it is, so save the model
            model_wrapper.save_state(
                os.path.join(logger.weights_dir, 'weights_best.pt')
            )

        # dont forget to write the samples
        logger.save_samples(model_wrapper.sample_generator(),           f'{current_epoch}_static.png')
        logger.save_samples(model_wrapper.sample_generator(batch_size), f'{current_epoch}_random.png')

        # write stats
        logger.write()
        print("|")

    print("| Training is Complete !")

    # it is, so save the model
    model_wrapper.save_state(
        os.path.join(logger.weights_dir, 'weights_last.pt')
    )

    print("| Wrapping Up")












