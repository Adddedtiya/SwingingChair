import os
import argparse

import torch
import random
import numpy as np

from data.dataset_reconstruction import ReconstructionDataset
from helpers.dictonary_tracker   import TrackerAndLogger
from helpers.speed_tracker       import TimeTracker
from helpers.wrapper_maet        import WrapperMAET
from models.model_siaq           import LigweightAutoencoderK512
from models.model_murt           import SimpleMurq

# Deterministic Algorithms
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("## Training QAE ##")

    # setup args
    parser = argparse.ArgumentParser(description = "VAE-GAN Training configuration")
    parser.add_argument('--total_epochs', type = int, default = 512)
    parser.add_argument('--batch_size',   type = int, default = 16)
    parser.add_argument('--load_threads', type = int, default = 4)
    parser.add_argument('--dataset_root', type = str, default = '')
    parser.add_argument('--name'        , type = str, default = '')
    parser.add_argument('--memory_cache', action = 'store_true')
    parser.add_argument('--color',        action = 'store_true')
    parser.add_argument('--image_size',   type = int, default = 512)
    parser.add_argument('--ae_weight',    type = str)
    parser.add_argument('--mask_ratio',   type = float, default = 0.4)
    
    # Prase the Arguemnts
    parsed_args  = parser.parse_args()    
    total_epochs : int   = parsed_args.total_epochs
    batch_size   : int   = parsed_args.batch_size
    load_threads : int   = parsed_args.load_threads
    dataset_root : str   = parsed_args.dataset_root
    exp_name     : str   = parsed_args.name
    memory_cache : bool  = parsed_args.memory_cache
    use_colour   : bool  = parsed_args.color
    image_size   : int   = parsed_args.image_size
    ae_weight    : str   = parsed_args.ae_weight
    mask_ratio   : float = parsed_args.mask_ratio

    print("| Pytorch Model Training !")
    print("| Total Epoch :", total_epochs)
    print("| Batch Size  :", batch_size)
    print("| Workers     :", load_threads)
    print("| Device      :", device)
    print("| Size        :", image_size)
    print("| Name        :", exp_name)
    print("| Mem-Only    :", memory_cache)

    # create Trackers
    logger = TrackerAndLogger('./runs', exp_name, metric_to_track = 'top_1')

    # for model and dataloader
    colour_channels = 3 if use_colour else 1

    # create the model in the wrapper 
    autoencoder_model = LigweightAutoencoderK512(
        input_channels  = colour_channels,
        output_channels = colour_channels
    )
    ae_weights = torch.load(ae_weight, map_location = device)
    autoencoder_model.load_state_dict(ae_weights['autoencoder'])

    murq_model = SimpleMurq(
        input_size     = 8,
        length         = 1024,
        output_classes = 64,
        latent_size    = 768,
        depth          = 13,
        heads          = 12,
        ff_dim         = 1024
    )

    model_wrapper = WrapperMAET(
        autoencoder  = autoencoder_model,
        model        = murq_model,
        masked_ratio = mask_ratio,
        device       = device
    )

    # Create Dataset
    train_dataset = ReconstructionDataset(dataset_root, 'train', memory_cache, use_colour, image_size)
    eval_dataset  = ReconstructionDataset(dataset_root, 'eval',  memory_cache, use_colour, image_size)
    # test_dataset  = ReconstructionDataset(dataset_root, 'test',  memory_cache, use_colour, image_size)

    # create the dataloaders
    loader_eval  = eval_dataset.create_dataloader(batch_size, load_threads, device, shuffle = False)
    # loader_test  = test_dataset.create_dataloader(batch_size, load_threads, device)
    loader_train = train_dataset.create_dataloader(batch_size, load_threads, device)

    print("| Setup Complete Start Training !")    
    for current_epoch in range(total_epochs):

        TimeTracker.start_clock()
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
        
        # dont forget to write the samples
        logger.save_samples(model_wrapper.sample_generator(loader_eval), f'{current_epoch}_eval_sample.png', nrow = 1)

        # write stats
        logger.write()

        # plot the ssim over epochs
        logger.combined_plot(
            training_keys   = ['cross_entropy_loss'],
            evaluation_keys = ['cross_entropy_loss'],
            title = 'Cross Entropy over Epochs',
            fname = 'cross_plot.png'
        )

        # compute the ETA here !
        TimeTracker.stop_clock()
        remaning_epoch  = total_epochs - (current_epoch + 1)
        time_estimation = TimeTracker.estimate_time(remaning_epoch) 
        print("|", time_estimation)
        print("|", flush = True)

    print("| Training is Complete !")

    # it is, so save the model
    model_wrapper.save_state(
        os.path.join(logger.weights_dir, 'weights_last.pt')
    )

    print("| Wrapping Up")