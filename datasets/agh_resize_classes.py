import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import teeth_instance, teeth_semantic
from processing.prepare_tooth_fairy_data import prepare_toothfairy
from monai.transforms import Resize
from datasets.init import init_agh, validation_spliter, init_tooth_fairy

class AGH_Resize_Classes(Dataset):
    
    def __init__(self, data_directory:str = 'D:/db1_v1.0',
                        input_size = (128, 128, 128), 
                        augmentation = False,
                        crop = False,
                        validation_split = 0.0, 
                        validation_mode = False
                    ):
        
        self.data_directory = data_directory
        self.input_size = input_size
        self.augmentation = augmentation
        self.crop = crop
        self.validation_split = validation_split
        self.validation_mode = validation_mode

        if self.data_directory.endswith('ToothFairy3_v1.0'):
            self.lista = init_tooth_fairy(self.data_directory)
        else:
            self.lista = init_agh(self.data_directory)
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
                       
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label, path_label_final = paths

        label_weak = load_itk(path_label)
        label = load_itk(path_label_final)

        if self.data_directory.endswith('ToothFairy3_v1.0'):
            label_weak = teeth_semantic(label_weak)
            label = teeth_instance(label)
        else:
            label_weak = np.flip(label_weak, axis=(0,2))
            label = np.flip(label, axis=(0, 2))

        if self.crop:
            label_weak, label = crop_to_mask(label_weak, label, padding=5)

        label_weak = label_weak.astype(np.float32)
        label = label.astype(np.float32)

        down = Resize(spatial_size = self.input_size, 
                      size_mode='all', 
                      mode="trilinear", 
                      align_corners=None, 
                      anti_aliasing=True, 
                      anti_aliasing_sigma=None,
                      dtype=np.float32, 
                      lazy=False)
        
        label_weak = down.__call__(label_weak[None, ...])
        label = down.__call__(label[None, ...])[0]

        return label_weak, label
    

