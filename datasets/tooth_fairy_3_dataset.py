import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy
from datasets.init import init_tooth_fairy, validation_spliter

class ToothFairy3_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'D:/ToothFairy3',
                        input_size = (96, 96, 96), 
                        remove_ct_rings:bool = False, 
                        clamp:tuple[int, int] = None, 
                        compression_function = fix_tooth_fairy_classes,
                        normalization = norm_to_0_1,
                        augmentation = False,
                        channels = [], 
                        num_samples:int = 4, 
                        validation_split = 0.0, 
                        validation_mode = False
                    ):
        
        self.data_directory = data_directory
        self.remove_ct_rings = remove_ct_rings
        self.clamp = clamp
        self.input_size = input_size
        self.compression_function = compression_function
        self.normalization = normalization
        self.augmentation = augmentation
        self.channels = channels
        self.num_samples = num_samples
        self.validation_split = validation_split
        self.validation_mode = validation_mode

        self.lista = init_tooth_fairy(self.data_directory)
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
 
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label, path_label_2 = paths

        image = load_itk(path_cbct, clamp = self.clamp)
        mask = load_itk(path_label)

        mask = self.compression_function(mask)

        image, mask = crop_to_mask(image, mask, padding=5)

        assert image.shape == mask.shape

        image, mask = prepare_toothfairy(image, mask, self.remove_ct_rings, self.input_size, self.normalization, self.augmentation, self.channels)

        return image, mask
    

