import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy


class AGH_Class_Segmentor_Dataset(Dataset):
    
    def __init__(self, data_directory:str = 'D:/db1_v1.0',
                        input_size = (96, 96, 96), 
                        remove_ct_rings:bool = False, 
                        clamp: tuple[int, int] = None, 
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
        self.normalization = normalization
        self.augmentation = augmentation
        self.channels = channels
        self.validation_split = validation_split
        self.validation_mode = validation_mode

        self.lista = []

        self.path_cbct = os.path.join(self.data_directory, 'db1_v1.0_nifti')
        self.path_labels = os.path.join(self.data_directory, 'db1_v1.0_nifti_teeth_masks_single_cls_weak')
        self.path_labels_final = os.path.join(self.data_directory, 'db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak')
       
        paths_cbct = sorted(os.listdir(self.path_cbct), key=lambda x: int(x))    
        files_labels = sorted(os.listdir(self.path_labels), key=lambda x: int(x))
        files_labels_final = sorted(os.listdir(self.path_labels_final), key=lambda x: int(x))

        paths_cbct = paths_cbct[:len(files_labels)]
        
        for file_cbct, file_label, file_label_final in zip(paths_cbct, files_labels, files_labels_final):
            
            file_cbct = os.path.join(self.path_cbct, file_cbct, 'image.nii.gz')
            file_label = os.path.join(self.path_labels, file_label)
            file_label_final = os.path.join(self.path_labels, file_label_final)    
            self.lista.append((file_cbct, file_label, file_label_final))

        split_index = int(len(self.lista) * self.validation_split)
        
        if validation_mode:
            self.lista = self.lista[:split_index]
        else:
            self.lista = self.lista[split_index:]
             
                    
    def __len__(self):

        return len(self.lista)


    def __getitem__(self, index): 

        paths = self.lista[index]
        path_cbct, path_label, path_label_final = paths

        #image = load_itk(path_cbct, clamp = self.clamp)
        image = load_itk(path_label)
        mask = load_itk(path_label_final)

        image, mask = crop_to_mask(image, mask, padding=5)

        assert image.shape == mask.shape

        image, mask, coords = prepare_toothfairy(image, mask, self.remove_ct_rings, self.input_size, self.normalization, self.augmentation, self.channels, coords=True)

        #add positional layers
        x = np.ones(mask.shape, dtype = np.float32) * coords[0]
        y = np.ones(mask.shape, dtype = np.float32) * coords[1]
        z = np.ones(mask.shape, dtype = np.float32) * coords[2]

        image = np.concatenate((image, x, y, z), axis = 0)

        return image, mask, coords
    

