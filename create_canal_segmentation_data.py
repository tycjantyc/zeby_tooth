import os
import numpy as np
from processing.utils import load_itk, crop_to_mask
from processing.tooth_fairy_classes import teeth_canal_instance


def create_canal_data(data_dir = 'D:/ToothFairy3', padding = 4, input_size = 64):

    path_images = os.path.join(data_dir, 'imagesTr')
    path_labels = os.path.join(data_dir, 'labelsTr')

    paths_images = [os.path.join(path_images, i) for i in os.listdir(path_images)]
    paths_labels = [os.path.join(path_labels, i) for i in os.listdir(path_labels)]

    if os.path.isdir('data/snippets_img'):
        pass
    else:
        os.mkdir('data/snippets_img')
        os.mkdir('data/snippets_mask')

    global_num = 0

    for file_img, file_lab in zip(paths_images, paths_labels):
        
        image = load_itk(file_img)
        mask = load_itk(file_lab)

        mask = teeth_canal_instance(mask)

        for teeth_num in np.unique(mask)[1:]:
            if teeth_num > 32:
                continue
            
            temp_mask = np.zeros(mask.shape)
            temp_mask[(mask != teeth_num) & (mask != teeth_num+32)] = 0
            temp_mask[mask == teeth_num] = 1
            temp_mask[mask == teeth_num + 32] = 2
            
            image_snipped, mask_snipped = crop_to_mask(image, temp_mask, padding=padding, input=input_size)
            
            mask_snipped[mask_snipped==1] = 0
            mask_snipped[mask_snipped==2] = 1

            print(mask_snipped.min(), mask_snipped.max())
            print(image_snipped.shape)
            
            if image_snipped.shape[0] > 110:
                continue

            np.save(file=f'data/snippets_img/{global_num}.npz', arr=image_snipped)
            np.save(file=f'data/snippets_mask/{global_num}.npz', arr=mask_snipped)


if __name__ == "__main__":
    create_canal_data(input_size=10)