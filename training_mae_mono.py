import os
import argparse

import torch
import random
import numpy as np

from data.dataset_reconstruction import ReconstructionDataset
from helpers.dictonary_tracker   import TrackerAndLogger
from helpers.wrapper_mae         import WrapperMAE
from models.model_mae            import MonoMAE

# Deterministic Algorithms
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("## Training SAE ##")

    # setup args
    parser = argparse.ArgumentParser(description = "MAE Training configuration")
    parser.add_argument('--total_epochs', type = int, default = 512)
    parser.add_argument('--batch_size',   type = int, default = 16)
    parser.add_argument('--load_threads', type = int, default = 4)
    parser.add_argument('--latent_size',  type = int, default = 3072)
    parser.add_argument('--dataset_root', type = str, default = '')
    parser.add_argument('--name'        , type = str, default = '')
    parser.add_argument('--memory_cache', action = 'store_true')
    parser.add_argument('--color',        action = 'store_true')
    
    # MAE specific options
    parser.add_argument('--depth',             type = int, default = 27)
    parser.add_argument('--transfomer_heads',  type = int, default = 13)
    parser.add_argument('--transfomer_ffdim',  type = int, default = 2048)
    parser.add_argument('--mae_visible_ratio', type = float, default = 0.3)

    # Prase the Arguemnts
    parsed_args  = parser.parse_args()    
    total_epochs : int  = parsed_args.total_epochs
    batch_size   : int  = parsed_args.batch_size
    load_threads : int  = parsed_args.load_threads
    latent_size  : int  = parsed_args.latent_size
    dataset_root : str  = parsed_args.dataset_root
    exp_name     : str  = parsed_args.name
    memory_cache : bool = parsed_args.memory_cache
    use_colour   : bool = parsed_args.color

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

    # for model and dataloader
    colour_channels = 3 if use_colour else 1

    # create the model in the wrapper 
    autoencoder_model = MonoMAE(
        input_channels  = colour_channels,
        output_channels = colour_channels,
        image_size      = 128,
        patch_size      = 16,
        latent_size     = latent_size,
        depth           = parsed_args.depth,
        heads           = parsed_args.transfomer_heads,
        ff_dim          = parsed_args.transfomer_ffdim
    )
    model_wrapper = WrapperMAE(
        autoencoder   = autoencoder_model,
        latent_size   = latent_size,
        visible_ratio = parsed_args.mae_visible_ratio,
        device        = device
    )

    # Create Dataset
    train_dataset = ReconstructionDataset(dataset_root, 'train', memory_cache, use_colour)
    eval_dataset  = ReconstructionDataset(dataset_root, 'eval',  memory_cache, use_colour)
    test_dataset  = ReconstructionDataset(dataset_root, 'test',  memory_cache, use_colour)

    # create the dataloaders
    loader_eval  = eval_dataset.create_dataloader(batch_size, load_threads, device, shuffle = False)
    loader_test  = test_dataset.create_dataloader(batch_size, load_threads, device)
    loader_train = train_dataset.create_dataloader(batch_size, load_threads, device)

    print("| Setup Complete Start Training !")    
    for current_epoch in range(total_epochs):

        print(f"| Current Epoch {current_epoch + 1}/{total_epochs}")
        
        # Train the Model For a single epoch
        train_stats = model_wrapper.train_single_epoch(loader_train)

        # Evaluate the model
        print("| Training Complete, Evaluating...")
        eval_stats = model_wrapper.evaluate_single_epoch(loader_eval)

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
        
        # save the latest model too
        # model_wrapper.save_state(
        #     os.path.join(logger.weights_dir, 'weights_latest.pt')
        # )

        # dont forget to write the samples
        logger.save_samples(model_wrapper.sample_generator(loader_eval), f'{current_epoch}_random.png', nrow = 1)

        # write stats
        logger.write()

        # plot the ssim over epochs
        logger.combined_plot(
            training_keys   = ['ssim'],
            evaluation_keys = ['ssim'],
            title = 'SSIM over Epochs',
            fname = 'ssim_plot.png'
        )

        print("|")

    print("| Training is Complete !")

    # it is, so save the model
    model_wrapper.save_state(
        os.path.join(logger.weights_dir, 'weights_last.pt')
    )

    print("| Wrapping Up")