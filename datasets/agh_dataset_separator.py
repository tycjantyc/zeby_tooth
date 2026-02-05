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

def most_frequent_nonzero(arr):
    
    unique, counts = np.unique(arr, return_counts=True)

    print(unique, counts)

    # Sort by frequency (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_unique = unique[sorted_indices]

    # If top value is 0, pick next one
    if sorted_unique[0].item() == 0 and len(sorted_unique) > 1:
        return sorted_unique[1].item()
    else:
        return sorted_unique[0].item()

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

        if self.data_directory.endswith('ToothFairy3_v1.0'):
            self.lista = init_tooth_fairy(self.data_directory)
        else:
            self.lista = init_agh(self.data_directory)
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
        self.model_1 = build_class_segmentor_model(inchannels=1, outchannels=1, depths=[32, 64, 128, 256, 384])
        self.model_1.load_state_dict(load('models/weights/agh_centroid_best_s2.pt', weights_only=True))            
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_helper, path_label = paths

        image = load_itk(path_helper)
        mask = load_itk(path_label)

        centers = instance_teeth_segment(self.model_1, image, return_file = False)
        centers = np.flip(centers, axis = (0, 2))    
        guess = np.random.choice(np.unique(centers))

        image, _  = crop_to_mask_v2(image, (centers == guess)*1)
        mask, _ = crop_to_mask_v2(mask, (centers == guess)*1)

        num = most_frequent_nonzero(mask[40:88, 40:88, 40:88])
        
        if num == 0:
            num = most_frequent_nonzero(mask[20:108, 30:98, 30:98])
    
        mask[mask==num] = 100
        mask[(mask>0) & (mask<50)] = 1
        mask[mask==100] = 2

        return image, mask
    

