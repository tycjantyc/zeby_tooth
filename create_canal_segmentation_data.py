import os
import numpy as np
from processing.utils import load_itk, crop_to_mask, norm_to_0_1
from processing.tooth_fairy_classes import teeth_canal_instance
import argparse
from create_canal_segmentation_data_agh import create_canal_data_agh
from tqdm import tqdm
from monai.transforms import Spacing

def create_canal_data(data_dir = 'D:/ToothFairy3', padding = 4, input_size = 64):

    path_images = os.path.join(data_dir, 'imagesTr')
    path_labels = os.path.join(data_dir, 'labelsTr')

    paths_images = [os.path.join(path_images, i) for i in os.listdir(path_images)]
    paths_labels = [os.path.join(path_labels, i) for i in os.listdir(path_labels)]

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

    for number_image, (file_img, file_lab) in tqdm(enumerate(zip(paths_images, paths_labels))):
        
        number_image += 1

        image = load_itk(file_img)
        mask = load_itk(file_lab)
        mask = teeth_canal_instance(mask)

        for teeth_num in np.unique(mask)[1:]:
            if teeth_num > 32:
                continue
            
            global_num+=1

            temp_mask = np.zeros(mask.shape)
            temp_mask[(mask != teeth_num) & (mask != teeth_num+32)] = 0
            temp_mask[mask == teeth_num] = 1
            temp_mask[mask == teeth_num + 32] = 2
            
            image_snipped, mask_snipped = crop_to_mask(image, temp_mask, padding=padding, input=1)

            print(mask_snipped.min(), mask_snipped.max())
            print(image_snipped.shape)

            rescale = Spacing(pixdim=(2/3, 2/3, 2/3), mode = 'bilinear')

            image_snipped = rescale(image_snipped)
            mask_snipped = rescale(mask_snipped)

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
    arg_parser.add_argument('--db_dir', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--padding', type=int, required=False, default=5)
    arg_parser.add_argument('--input', type=float, required=False, default=96)
    args = arg_parser.parse_args()

    if args.db_dir.endswith('ToothFairy3_v1.0') or args.db_dir.endswith('ToothFairy3'):
        create_canal_data(data_dir=args.db_dir, padding=args.padding, input_size=args.input)
    else:
        create_canal_data_agh(data_dir=args.db_dir, padding=args.padding, input_size=args.input)

    