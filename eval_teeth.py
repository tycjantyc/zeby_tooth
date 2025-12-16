from evaluation.evaluate_canals_agh_better import eval
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
import argparse
import os
import numpy as np
from processing.utils import load_itk
from monai.losses import DiceLoss
import numpy as np

def dice_score_multiclass(y_true, y_pred, ignore_background=True, eps=1e-6):

    classes = list(set(np.append(np.unique(y_true), np.unique(y_pred))))
    if ignore_background:
        classes.remove(0)
   
    dice_per_class = {}

    for cls in classes:
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)

        intersection = np.sum(y_true_cls & y_pred_cls)
        union = np.sum(y_true_cls) + np.sum(y_pred_cls)

        if union == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection + eps) / (union + eps)

        dice_per_class[cls] = dice

    mean_dice = np.mean(list(dice_per_class.values()))

    return mean_dice #, dice_per_class


def num_extractor_help(path):
    return int(path.split('_')[-1].split('.')[0])

def main(data_path):

    pred_path = 'evaluation/eval_results_jenkins/'
    dice_scores = []

    for i in os.listdir(pred_path):

        if i.startswith('.'):
            continue

        pred_path_canal = os.path.join(pred_path, i)
        gt_path_canal = os.path.join(data_path, 'db1_v1.0','db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak',f'{num_extractor_help(pred_path_canal)}.nii.gz')
        
        print(gt_path_canal)

        dice_score = dice_score_multiclass(load_itk(pred_path_canal), load_itk(gt_path_canal))
        dice_scores.append(dice_score)
        print(f"Dice score: {dice_score}")

    print(f"Mean dice score: {np.mean(dice_scores)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=False, default='D:/ToothFairy3')
    
    args = arg_parser.parse_args()

    main(args.data_path)

    