import os
from monai.transforms import Resize
from processing.utils import load_itk, crop_to_mask
import numpy as np
from tqdm import tqdm

def resize_genrator(data_dir = 'D:/db1_v1.0'):

    if os.path.isdir('data'):
        pass
    else:
        os.mkdir('data')

    if os.path.isdir('data/snippets_img'):
        pass
    else:
        os.mkdir('data/snippets_img')
        os.mkdir('data/snippets_mask')


    path_images = os.path.join(data_dir, 'db1_v1.0_nifti')
    path_helper = os.path.join(data_dir, 'db1_v1.0_nifti_teeth_masks_single_cls_weak')
    path_labels = os.path.join(data_dir, 'db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak')

    paths_images = [os.path.join(path_images, i, 'image.nii.gz') for i in sorted(os.listdir(path_images), key=lambda x: int(x))]
    paths_helpers = [os.path.join(path_helper, i) for i in sorted(os.listdir(path_helper), key=lambda x: int(x.split('.')[0]))]
    paths_labels = [os.path.join(path_labels, i) for i in sorted(os.listdir(path_labels), key=lambda x: int(x.split('.')[0]))]

    paths_images = paths_images[1: len(path_labels)+1]
    paths_helpers = paths_helpers[: len(path_labels)]

    global_num = 0

    for path_weak, path_label in tqdm(zip(paths_helpers, paths_labels)):
        
        label_weak = load_itk(path_weak)
        label = load_itk(path_label)

        print(label_weak.shape)
        print(label.shape)

        label_weak, label = crop_to_mask(label_weak, label, padding=5)

        print(label_weak.shape)
        print(label.shape)

        down = Resize(spatial_size = (128, 128, 128), 
                      size_mode='all', 
                      mode="trilinear", 
                      align_corners=None, 
                      anti_aliasing=True, 
                      anti_aliasing_sigma=None,
                      dtype=np.float16, 
                      lazy=False)
        
        label_weak = down.__call__(label_weak[None, ...])
        label = down.__call__(label[None, ...])

        global_num += 1 
        
        np.save(file=f'data/snippets_img/{global_num}.npy', arr=label_weak)
        np.save(file=f'data/snippets_mask/{global_num}.npy', arr=label)

            
if __name__ == "__main__":
    resize_genrator()