import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy


class ToothFairy3_Class_Segmentor_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'D:/ToothFairy3', 
                        clamp:tuple[int, int] = (-1000, 3000), 
                        compression_function = fix_tooth_fairy_classes,
                        normalization = norm_to_0_1,
                        augmentation = False,
                        channels = []
                    ):
        
        self.data_directory = data_directory
        self.clamp = clamp
        self.compression_function = compression_function
        self.normalization = normalization
        self.augmentation = augmentation
        self.channels = channels

        self.lista = []

        self.path_cbct = os.path.join(self.data_directory, 'imagesTr')
        self.path_labels = os.path.join(self.data_directory, 'labelsTr')
       
        files_cbct = sorted(os.listdir(self.path_cbct))
        files_labels = sorted(os.listdir(self.path_labels))
         
        for file_cbct, file_label in zip(files_cbct, files_labels):
            
            file_cbct = os.path.join(self.path_cbct, file_cbct)
            file_label = os.path.join(self.path_labels, file_label)
            
            self.lista.append((file_cbct, file_label))
             
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        image = load_itk(path_cbct, clamp = self.clamp)
        mask_org = load_itk(path_label)

        mask = fix_tooth_fairy_classes(mask_org)

        mask = self.compression_function(mask_org)

        image = self.normalization(image)
        image, mask = prepare_toothfairy(image, mask, self.remove_ct_rings, self.input_size, self.normalization, self.augmentation, self.channels)

        return image, mask
    

