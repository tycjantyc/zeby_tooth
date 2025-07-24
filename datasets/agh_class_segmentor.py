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
                        normalization = lambda x:x,
                        augmentation = False,
                        channels = [], 
                        validation_split = 0.0, 
                        validation_mode = False,
                        encode_coords = True
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
        self.encode_coords = encode_coords

        self.lista = []

        self.path_cbct = os.path.join(self.data_directory, 'db1_v1.0_nifti')
        self.path_labels = os.path.join(self.data_directory, 'db1_v1.0_nifti_teeth_masks_single_cls_weak')
        self.path_labels_final = os.path.join(self.data_directory, 'db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak')
       
        paths_cbct = sorted(os.listdir(self.path_cbct), key=lambda x: int(x.split('.')[0]))   
        files_labels = sorted(os.listdir(self.path_labels), key=lambda x: int(x.split('.')[0]))
        files_labels_final = sorted(os.listdir(self.path_labels_final), key=lambda x: int(x.split('.')[0]))

        paths_cbct = paths_cbct[1:len(files_labels)+1]
        
        for file_cbct, file_label, file_label_final in zip(paths_cbct, files_labels, files_labels_final):
            
            file_cbct = os.path.join(self.path_cbct, file_cbct, 'image.nii.gz')
            file_label = os.path.join(self.path_labels, file_label)
            file_label_final = os.path.join(self.path_labels_final, file_label_final)    
            self.lista.append((file_cbct, file_label, file_label_final))

        self.lista = self.lista[:1]

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
        image = load_itk(path_cbct)
        mask = load_itk(path_label_final)

        image, mask = crop_to_mask(image, mask, padding=5)

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
            image = np.stack([image, grid_x, grid_y, grid_z], axis=0)
            return image.astype(np.float32), mask
        
        else:
            return image, mask, coords
    

