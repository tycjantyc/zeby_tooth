import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes, teeth_instance, teeth_semantic
from processing.prepare_tooth_fairy_data import prepare_toothfairy
from datasets.init import init_tooth_fairy, validation_spliter

class ToothFairy3_Class_Segmentor_Dataset(Dataset):
    
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
                        validation_mode = False,
                        encode_coords = True
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
        self.encode_coords = encode_coords

        self.lista = init_tooth_fairy(self.data_directory)
        self.lista = validation_spliter(self.lista, validation_mode=self.validation_mode, validation_split=self.validation_split)
 
        
        
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label = paths

        #image = load_itk(path_cbct, clamp = self.clamp)
        image = load_itk(path_label)
        mask = load_itk(path_label)

        image = teeth_semantic(image)
        mask = teeth_instance(mask)

        image, mask = crop_to_mask(image, mask, padding=48, input = self.input_size[0])

        assert image.shape == mask.shape

        image, mask, coords = prepare_toothfairy(image, mask, self.remove_ct_rings, self.input_size, self.normalization, self.augmentation, self.channels, coords=True)

        zz, yy, xx = coords

        z0, z1 = zz
        y0, y1 = yy
        x0, x1 = xx

        #add positional layers
        z = np.linspace(z0, z1, 96)
        y = np.linspace(y0, y1, 96)
        x = np.linspace(x0, x1, 96)

        grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')  
          
        if self.encode_coords:
            image = np.stack([image[0], grid_x, grid_y, grid_z], axis=0)
            return image.astype(np.float32), mask
        
        else:
            return image, mask, coords
    

