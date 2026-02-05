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
from datasets.agh_dataset_separator import most_frequent_nonzero
import argparse
from tqdm import tqdm

def main(data_directory = 'D:/db1_v1.0'):

    if os.path.isdir('data'):
        pass
    else:
        os.mkdir('data')

    if os.path.isdir('data/snippets_img'):
        pass
    else:
        os.mkdir('data/snippets_img')
        os.mkdir('data/snippets_mask')

    lista = init_agh(data_directory)

    lista = lista[20:]
    print(lista)

    model_1 = build_class_segmentor_model(inchannels=1, outchannels=1, depths=[32, 64, 128, 256, 384])
    model_1.load_state_dict(load('models/weights/agh_centroid_best_s2.pt', weights_only=True))   
        
    for idx, path in tqdm(enumerate(lista)):
    
        path_cbct, path_helper, path_label = path

        image = load_itk(path_helper)
        mask = load_itk(path_label)

        centers = instance_teeth_segment(model_1, image, return_file = False)
        centers = np.flip(centers, axis = (0, 2))    
    
        for guess in np.unique(centers):

            if guess == 0:
                continue

            print(np.unique((centers == guess)*1))
            image_temp, _  = crop_to_mask_v2(image, (centers == guess)*1)
            mask_temp, _ = crop_to_mask_v2(mask, (centers == guess)*1)

            print(np.unique(mask_temp), mask_temp.shape)
            num = most_frequent_nonzero(mask_temp[40:88, 40:88, 40:88])

            print(num)
            
            if num == 0:
                num = most_frequent_nonzero(mask_temp[20:108, 30:98, 30:98])
        
            mask_temp[mask_temp==num] = 100
            mask_temp[(mask_temp>0) & (mask_temp<50)] = 1
            mask_temp[mask_temp==100] = 2

            np.save(file=f'data/snippets_img/{idx+21}_{guess}.npy', arr=image_temp.astype(np.uint16))
            np.save(file=f'data/snippets_mask/{idx+21}_{guess}.npy', arr=mask_temp.astype(np.uint8))

            
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--db_dir', type=str, required=False, default='D:/ToothFairy3')
    args = arg_parser.parse_args()

    main(data_directory=args.db_dir)


   