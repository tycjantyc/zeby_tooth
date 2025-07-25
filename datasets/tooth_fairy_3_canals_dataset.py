import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import embed_in_center
from processing.data_augmentation import *

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

        self.lista = []

        self.path_cbct = os.path.join(self.data_directory, 'snippets_img')
        self.path_labels = os.path.join(self.data_directory, 'snippets_mask')

        files_cbct = sorted(os.listdir(self.path_cbct))
        files_labels = sorted(os.listdir(self.path_labels))
         
        for file_cbct, file_label in zip(files_cbct, files_labels):
            
            file_cbct = os.path.join(self.path_cbct, file_cbct)
            file_label = os.path.join(self.path_labels, file_label)
            self.lista.append((file_cbct, file_label))

        split_index = int(len(self.lista) * self.validation_split)
        
        if validation_mode:
            self.lista = self.lista[:split_index]
        else:
            self.lista = self.lista[split_index:]
             
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        image = np.load(path_cbct)
        mask = np.load(path_label)

        mask[mask==1] == 0
        mask[mask==2] == 1

        assert image.shape == mask.shape
        
        if image.shape[0] < self.input_size[0] and image.shape[1] < self.input_size[1]: 
            image = embed_in_center(image, self.input_size[0])
        
        else:
            temp = np.zeros(self.input_size)
            temp[0: min(image.shape[0], self.input_size[0]), 0: min(image.shape[1], self.input_size[1]), 0: min(image.shape[2], self.input_size[2])] = image[0: min(image.shape[0], self.input_size[0]), 0: min(image.shape[1], self.input_size[1]), 0: min(image.shape[2], self.input_size[2])]
            image = temp

        if self.augmentation:
            factory = AugmentationFactory()
            image, mask = factory.create(image, mask)

        return image.astype(np.float32), mask.astype(np.float32)
    

