import os
import numpy as np
from torch.utils.data import Dataset
from processing.utils import load_itk, norm_to_0_1, crop_to_mask
from processing.tooth_fairy_classes import fix_tooth_fairy_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy
from monai.transforms import Resize

class AGH_Resize_Classes(Dataset):
    
    def __init__(self, data_directory:str = 'D:/db1_v1.0',
                        input_size = (128, 128, 128), 
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

        label_weak = load_itk(path_label)
        label = load_itk(path_label_final)

        #label_weak, label = crop_to_mask(label_weak, label, padding=5)

        down = Resize(spatial_size = (128, 128, 128), 
                      size_mode='all', 
                      mode="trilinear", 
                      align_corners=None, 
                      anti_aliasing=True, 
                      anti_aliasing_sigma=None,
                      dtype=np.uint8, 
                      lazy=False)
        
        label_weak = down.__call__(label_weak[None, ...])
        label = down.__call__(label[None, ...])[0]

        return label_weak, label
    

