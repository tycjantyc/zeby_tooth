import numpy as np
import sys
import SimpleITK as sitk
import os
sys.path.append(os.getcwd().removesuffix('\\evaluation'))
print(sys.path)
import nibabel as nib
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from models.class_segmentor_model import build_class_segmentor_model
import torch as tc
import torch.nn as nn
from processing.tooth_fairy_classes import teeth_instance, only_canal_classes
from processing.utils import load_itk, crop_to_mask
from processing.utils import embed_in_center
from tqdm import tqdm
from monai.losses import DiceLoss

model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False, weight_path='../models/weights/tooth_fairy_3d_best_agh.pt')
model.eval()
model = model.cuda()

FILE = '9'
THRESHOLD = 0.5
LOSS = DiceLoss(include_background=True, sigmoid=False)
DATA = "agh" 
DATA = "toothfairy"

if DATA == "agh":
    file = f'D:/db1_v1.0/db1_v1.0_nifti/{FILE}/image.nii.gz'
    helper_path = f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{FILE}.nii.gz'
    ground_truth = f'D:/db1_v1.0/db1_v1.0_nifti_root_canal_masks_single_cls_ref/{FILE}.nii.gz'

elif DATA == "toothfairy":
    file = f'D:/ToothFairy3/imagesTr/ToothFairy3F_{int(FILE):03}_0000.nii.gz'
    helper_path = f'D:/ToothFairy3/labelsTr/ToothFairy3F_{int(FILE):03}.nii.gz'
    ground_truth = f'D:/ToothFairy3/labelsTr/ToothFairy3F_{int(FILE):03}.nii.gz'
    

image = load_itk(file)
helper = load_itk(helper_path)
ground_truth = load_itk(ground_truth)

if DATA == "toothfairy":
    helper = teeth_instance(helper)
    ground_truth = only_canal_classes(ground_truth)

canal_mask = np.zeros(image.shape)

for teeth_num in tqdm(np.unique(helper)[1:]):

    temp_mask = np.zeros(helper.shape)

    temp_mask[helper != teeth_num] = 0
    temp_mask[helper == teeth_num] = 1 

    image_snipped, mask_snipped, coords = crop_to_mask(image, temp_mask, padding=2, input=1, coordss=True)
    gt_snipped, _= crop_to_mask(ground_truth, temp_mask, padding=2, input=1, coordss=False)
    
    if ((image_snipped.shape[0] < 96) and (image_snipped.shape[1] < 96) and (image_snipped.shape[2] < 96)): 
        image_snip_2, cords_2 = embed_in_center(image_snipped, 96, coords=True)
        mask_snip_2 = embed_in_center(mask_snipped, 96, coords=False)
        gt_snip_2 = embed_in_center(gt_snipped, 96, coords=False)

        masked = image_snip_2[mask_snip_2>0]
        image_snip_2[mask_snip_2>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5)
        image_snip_2[mask_snip_2<1] = 0 

        image_snip_2 = tc.tensor(image_snip_2[None, None, ...], dtype = tc.float32).cuda()
        
        with tc.no_grad():
            output = nn.Sigmoid()(model(image_snip_2))

        dice_loss = LOSS(output, tc.tensor(gt_snip_2[None, None, ...], dtype=tc.float32).cuda())
        
        print(f"Teeth number {teeth_num} - Dice loss: {dice_loss.item()}")

        start, end = cords_2
        output = output.detach().cpu().numpy()[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        output[output > THRESHOLD] = 1
        output[output < THRESHOLD] = 0

        z_min, z_max, y_min, y_max, x_min, x_max = coords

        canal_mask[z_min:z_max, y_min:y_max, x_min:x_max] = output
        del output
        tc.cuda.empty_cache()
     
    else:   
            z_half = image_snipped.shape[0]//2
            image_snip_12, cords_1 = embed_in_center(image_snipped[:z_half], 96, coords=True)
            image_snip_22, cords_2 = embed_in_center(image_snipped[z_half:], 96, coords=True)
            
            mask_snip_12 = embed_in_center(mask_snipped[:z_half], 96, coords=False)
            mask_snip_22 = embed_in_center(mask_snipped[z_half:], 96, coords=False)

            gt_snip_12 = embed_in_center(gt_snipped[:z_half], 96, coords=False)
            gt_snip_22 = embed_in_center(gt_snipped[z_half:], 96, coords=False)

            masked = image_snip_12[mask_snip_12>0]
            image_snip_12[mask_snip_12>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5)
            image_snip_12[mask_snip_12<1] = 0 

            masked = image_snip_22[mask_snip_22>0]
            image_snip_22[mask_snip_22>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5)
            image_snip_22[mask_snip_22<1] = 0 

            output = image_snipped.copy()

            image_snip_12 = tc.tensor(image_snip_12, dtype = tc.float32).cuda()
            image_snip_12 = image_snip_12[None, None, ...]

            image_snip_22 = tc.tensor(image_snip_22, dtype = tc.float32).cuda()
            image_snip_22 = image_snip_22[None, None, ...]

            start1, end1 = cords_1
            start2, end2 = cords_2
            with tc.no_grad():
                pred = nn.Sigmoid()(model(image_snip_12))
            dice_loss = LOSS(pred, tc.tensor(gt_snip_12[None, None, ...], dtype=tc.float32).cuda())
            print(f"Teeth number {teeth_num} - Dice loss: {dice_loss}")
            output[:z_half] = pred.detach().cpu().numpy()[0, 0, start1[0]:end1[0], start1[1]:end1[1],start1[2]:end1[2]]

            with tc.no_grad():
                pred = nn.Sigmoid()(model(image_snip_22))
            dice_loss = LOSS(pred, tc.tensor(gt_snip_22[None, None, ...], dtype=tc.float32).cuda())
            print(f"Teeth number {teeth_num} - Dice loss: {dice_loss.item()}")
            output[z_half:] = pred.detach().cpu().numpy()[0, 0, start2[0]:end2[0], start2[1]:end2[1],start2[2]:end2[2]]
            
            output[output > THRESHOLD] = 1
            output[output < THRESHOLD] = 0

            z_min, z_max, y_min, y_max, x_min, x_max = coords
        
            canal_mask[z_min:z_max, y_min:y_max, x_min:x_max] = output

canal_mask[helper == 0] = 0
nifti_img = sitk.GetImageFromArray(canal_mask.astype(np.uint8))
original = sitk.ReadImage(helper_path)
nifti_img.CopyInformation(original)

if DATA == "toothfairy":
    sitk.WriteImage(nifti_img, f'../evaluation/evaluation_results/output_canals_mask_tf_{FILE}.nii.gz')
else:
    sitk.WriteImage(nifti_img, f'../evaluation/evaluation_results/output_canals_mask_{FILE}.nii.gz')

def dice(mask1, mask2):
    tp = np.count_nonzero((mask1+mask2) == 2)
    fp = np.count_nonzero((mask1+mask2) == 1)

    return 2*tp/(2*tp+fp)

print(f"Dice score: {dice(canal_mask, ground_truth)}")