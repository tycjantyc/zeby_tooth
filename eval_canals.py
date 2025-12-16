from evaluation.evaluate_canals_agh_better import eval
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
import argparse
import os
import numpy as np
from processing.utils import load_itk


def dice(mask1, mask2):
    tp = np.count_nonzero((mask1+mask2) == 2)
    fp = np.count_nonzero((mask1+mask2) == 1)
    return 2*tp/(2*tp+fp)

def num_extractor_help(path):
    return int(path.split('_')[-1].split('.')[0])

def main(data_path):

    pred_path = 'evaluation/eval_results_jenkins/'
    dice_scores = []

    for i in os.listdir(pred_path):

        if i.startswith('.'):
            continue

        pred_path_canal = os.path.join(pred_path, i)
        gt_path_canal = os.path.join(data_path, 'db1_v1.0','db1_v1.0_nifti_root_canal_masks_single_cls_ref',f'{num_extractor_help(pred_path_canal)}.nii.gz')
        
        print(gt_path_canal)

        dice_score = dice(load_itk(pred_path_canal), load_itk(gt_path_canal))
        dice_scores.append(dice_score)
        print(f"Dice score: {dice_score}")

    print(f"Mean dice score: {np.mean(dice_scores)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=False, default='D:/ToothFairy3')
    
    args = arg_parser.parse_args()

    main(args.data_path)

    