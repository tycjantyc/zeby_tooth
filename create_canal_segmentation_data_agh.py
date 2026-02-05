import os
import numpy as np
from processing.utils import load_itk, crop_to_mask, norm_to_0_1
from processing.tooth_fairy_classes import teeth_canal_instance
import argparse
from tqdm import tqdm


def create_canal_data_agh(data_dir = 'D:/db1_v1.0', padding = 4, input_size = 64):

    path_images = os.path.join(data_dir, 'db1_v1.0_nifti')
    path_helper = os.path.join(data_dir, 'db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak')
    path_labels = os.path.join(data_dir, 'db1_v1.0_nifti_root_canal_masks_single_cls_ref')

    paths_images = [os.path.join(path_images, i, 'image.nii.gz') for i in sorted(os.listdir(path_images), key=lambda x: int(x))]
    paths_helpers = [os.path.join(path_helper, i) for i in sorted(os.listdir(path_helper), key=lambda x: int(x.split('.')[0]))]
    paths_labels = [os.path.join(path_labels, i) for i in sorted(os.listdir(path_labels), key=lambda x: int(x.split('.')[0]))]

    paths_images = paths_images[1: len(path_labels)+1]
    paths_helpers = paths_helpers[: len(path_labels)]

    if os.path.isdir('data'):
        pass
    else:
        os.mkdir('data')

    if os.path.isdir('data/snippets_img'):
        pass
    else:
        os.mkdir('data/snippets_img')
        os.mkdir('data/snippets_mask')

    global_num = 0

    for file_img, file_help, file_lab in tqdm(zip(paths_images, paths_helpers, paths_labels)):
        
        number_image = int(file_help.split('/')[-1].split('.')[0])

        image = load_itk(file_img)
        helper = load_itk(file_help)
        mask = load_itk(file_lab)

        print(file_help, file_lab)

        print(np.unique(helper))
        print(mask.max())

        for teeth_num in np.unique(helper)[1:]:
            
            global_num+=1

            temp_mask = np.zeros(mask.shape)
            print(helper.shape)
            print(temp_mask.shape)
            temp_mask[helper != teeth_num] = 0
            temp_mask[helper == teeth_num] = 1 
            temp_mask[(helper == teeth_num) & (mask == 1)] = 2
            image_snipped, mask_snipped = crop_to_mask(image, temp_mask, padding=padding, input=1)

            print(mask_snipped.min(), mask_snipped.max())
            print(image_snipped.shape)

            if image_snipped.shape[0] > 96:

                z_half = image_snipped.shape[0]//2

                np.save(file=f'data/snippets_img/{number_image}_{teeth_num}_1.npy', arr=image_snipped[:z_half])
                np.save(file=f'data/snippets_mask/{number_image}_{teeth_num}_1.npy', arr=mask_snipped[:z_half])

                global_num += 1

                np.save(file=f'data/snippets_img/{number_image}_{teeth_num}_2.npy', arr=image_snipped[z_half:])
                np.save(file=f'data/snippets_mask/{number_image}_{teeth_num}_2.npy', arr=mask_snipped[z_half:])
            
            else:
                np.save(file=f'data/snippets_img/{number_image}_{teeth_num}.npy', arr=image_snipped)
                np.save(file=f'data/snippets_mask/{number_image}_{teeth_num}.npy', arr=mask_snipped)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--db_dir', type=str, required=False, default='D:/db1_v1.0')
    arg_parser.add_argument('--padding', type=int, required=False, default=5)
    arg_parser.add_argument('--input', type=float, required=False, default=96)
    args = arg_parser.parse_args()

    create_canal_data_agh(data_dir=args.db_dir, padding=args.padding, input_size=args.input)