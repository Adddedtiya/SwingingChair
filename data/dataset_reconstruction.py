import os
import torch
import torch.nn             as nn
import torch.nn.functional  as F

from torch.utils.data import DataLoader, Dataset
from einops import rearrange

import cv2             as cv
import albumentations  as A
import numpy           as np

from .image_directory import scan_directory

from tqdm import tqdm

class ReconstructionDataset(Dataset):
    def __init__(self, dataset_root : str, phase : str = 'train', memory : bool = False, color : bool = False):
        super().__init__()

        # for checking if color or grayscale
        self.color = color

        # get the image paths of your dataset;
        self.image_files_paths : list[str] = []

        # information
        self.memory      = memory
        self.subset      = str(phase).lower()
        self.is_training = bool(self.subset == 'train')

        # image shape
        self.img_load_size  = 96
        self.img_final_size = 64

        # get the image directory
        image_root_directory = os.path.join(dataset_root, self.subset)  
        image_root_directory = os.path.abspath(image_root_directory)

        # scan the directory for images
        self.image_files_paths += scan_directory(image_root_directory)

        # base augmentation procesing
        self.prepare_images = A.Compose([
            A.SmallestMaxSize(self.img_load_size, interpolation = cv.INTER_NEAREST),
        ])

        # image augmentation goes here !
        self.transform_image = A.Compose([
            A.Affine(scale = 1, translate_px = [-5, 5], rotate = [-50, 50], p = 1.0),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomCrop(self.img_final_size, self.img_final_size, p = 1.0),
        ]) if self.is_training else A.Compose([
            A.RandomCrop(self.img_final_size, self.img_final_size, p = 1.0)
        ])

        # load to memory if flag is enabled
        self.image_array_list : list[np.ndarray] = []
        if self.memory:
            print("> Load Files to Memory !")
            self.__load_to_memory()

        print(f"| Loaded '{self.subset}' with {len(self.image_files_paths)} | Cache : {len(self.image_array_list)}")

    def __load_to_memory(self) -> None:
        for fpath in tqdm(self.image_files_paths):
            image_array = self.__load_image_manual(fpath)
            self.image_array_list.append(image_array)

    def __load_image_file(self, fpath : str) -> np.ndarray:
        if self.color:
            # read the image as colour and to RGB
            original_image : np.ndarray = cv.imread(fpath, cv.IMREAD_COLOR)
            original_image : np.ndarray = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        else:
            # read the image as grayscale and expand color dim
            original_image : np.ndarray = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
            original_image : np.ndarray = rearrange(original_image, 'h w -> h w 1')
        
        # return the loaded image
        return original_image

    def __load_image_manual(self, fpath : str) -> np.ndarray:
        # read image 
        original_image = self.__load_image_file(fpath)

        # MinMax this time ?
        original_image : np.ndarray = original_image.astype(np.float32)
        original_image : np.ndarray = (original_image - original_image.min()) / np.ptp(original_image)

        original_image : np.ndarray = self.prepare_images(image = original_image)['image']
        original_image : np.ndarray = np.clip(original_image, 0, 1)
        original_image : np.ndarray = original_image.astype(np.float32)
        
        return original_image

    def __grab_image(self, index : int) -> np.ndarray:
        # grab from memory
        if self.memory:
            return self.image_array_list[index]
        
        # load the file and return
        file_path = self.image_files_paths[index]
        return self.__load_image_manual(file_path)

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files_paths)


    def __getitem__(self, index : int):

        # grab the image
        image_array = self.__grab_image(index)

        # instance augmentation
        image_array : np.ndarray = self.transform_image(image = image_array)['image']
        image_array : np.ndarray = np.clip(image_array, 0, 1)
        image_array : np.ndarray = image_array.astype(np.float32)
        image_array : np.ndarray = rearrange(image_array, 'h w c -> c h w')

        return image_array
    
    def create_dataloader(self, batch_size : int, total_workers : int, device : torch.device, shuffle : bool = True) -> DataLoader:

        # check variables and states
        total_workers = 0 if self.memory else total_workers
        using_gpu     = (str(device) == 'cuda')
        persistent    = total_workers > 0 

        return DataLoader(
            self,
            batch_size         = batch_size,
            shuffle            = shuffle,
            pin_memory         = using_gpu,
            num_workers        = total_workers,
            persistent_workers = persistent
        )