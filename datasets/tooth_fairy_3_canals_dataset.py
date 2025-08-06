import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import embed_in_center
from processing.data_augmentation import *
import SimpleITK as sitk
from datasets.init import validation_spliter, init_canal_snippets

class ToothFairy3_Canal_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'data',
                        input_size = (96, 96, 96),                 
                        augmentation = False,
                        validation_split = 0.0, 
                        validation_mode = False
                    ):
        
        self.data_directory = data_directory
        self.input_size = input_size
        self.augmentation = augmentation
        self.validation_split = validation_split
        self.validation_mode = validation_mode

        self.lista = init_canal_snippets(self.data_directory)
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
               
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        image = np.load(path_cbct).astype(np.float32)
        mask = np.load(path_label).astype(np.float32)

        masked = image[mask>0]
        if len(masked) > 0:
            image[mask>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5) #normalizacja dla lepszego kontrastu
        image[mask==0] = 0 #zalewanie wszystkiego co nie jest kanałem

        mask[mask<=1] = 0
        mask[mask>1] = 1

        # new = image.copy()
        # new = sitk.GetImageFromArray(new)

        # equalized = sitk.AdaptiveHistogramEqualization(image, alpha=1.0, beta=0.0, radius = 5)
        # new = sitk.GetArrayFromImage(equalized)
        
        if image.shape[0] < self.input_size[0] and image.shape[1] < self.input_size[1] and image.shape[2] < self.input_size[2]: 
            image = embed_in_center(image, self.input_size[0])
            mask = embed_in_center(mask, self.input_size[0])
        else:
            z, y, x = image.shape
            temp_img = np.zeros((96, 96, 96))
            temp_mask = np.zeros((96, 96, 96))
            temp_img[0:min(96, z), 0:min(96, y):, 0:min(96, x)] = image[0:min(96, z), 0:min(96, y):, 0:min(96, x)]
            temp_mask[0:min(96, z), 0:min(96, y):, 0:min(96, x)] = mask[0:min(96, z), 0:min(96, y):, 0:min(96, x)]

            image = temp_img
            mask = temp_mask

        if self.augmentation:
            factory = AugmentationFactory()
            image, mask = factory.create(image, mask)

        return image.astype(np.float32)[None, ...], mask.astype(np.float32)[None, ...]
    

