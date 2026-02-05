import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy
from datasets.init import init_agh, validation_spliter, init_tooth_fairy
from evaluation.test_centroid import instance_teeth_segment
from torch import load
from models.class_segmentor_model import build_class_segmentor_model
from processing.utils import crop_to_mask_v2
from processing.data_augmentation import *


class AGH_Teeth_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'D:/db1_v1.0',
                        input_size = (128, 128, 128), 
                        remove_ct_rings:bool = False, 
                        clamp: tuple[int, int] = None, 
                        compression_function = fix_tooth_fairy_classes,
                        normalization = norm_to_0_1,
                        augmentation = False,
                        channels = [], 
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
        self.validation_split = validation_split
        self.validation_mode = validation_mode


        self.lista = [(os.path.join('data/snippets_img', i), os.path.join('data/snippets_mask', j)) for i, j in zip(os.listdir('data/snippets_img'), os.listdir('data/snippets_mask'))]
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
               
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        image = np.load(path_cbct).astype(np.float32)
        mask = np.load(path_label).astype(np.float32)

        if self.augmentation:
            factory = AugmentationFactory()
            image, mask = factory.create(image, mask)
        
        return image, mask
    

