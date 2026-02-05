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
from processing.utils import infill_label


def eval(model, file_cbct, file_instance_seg, file_gt, out, infill = False):
    
    THRESHOLD = 0.5
    LOSS = DiceLoss(include_background=True, sigmoid=False)
    
    image = load_itk(file_cbct)
    helper = load_itk(file_instance_seg)
    if file_gt is not None:
        ground_truth = load_itk(file_gt)

    print(image.shape)
    print(helper.shape)

    if infill:
        for teeth_num in tqdm(np.unique(helper)[1:]):
            helper = infill_label(helper, target=teeth_num, closing_iters=4, background=0, method='infill')

    canal_mask = np.zeros(image.shape)

    for teeth_num in tqdm(np.unique(helper)[1:]):

        temp_mask = np.zeros(helper.shape)

        temp_mask[helper != teeth_num] = 0
        temp_mask[helper == teeth_num] = 1

        image_snipped, mask_snipped, coords = crop_to_mask(image, temp_mask, padding=0, input=1, coordss=True)
        if file_gt is not None:
            gt_snipped, _= crop_to_mask(ground_truth, temp_mask, padding=0, input=1, coordss=False)
        

        if ((image_snipped.shape[0] < 96) and (image_snipped.shape[1] < 96) and (image_snipped.shape[2] < 96)): 
            image_snip_2, cords_2 = embed_in_center(image_snipped, 96, coords=True)
            mask_snip_2 = embed_in_center(mask_snipped, 96, coords=False)
            if file_gt is not None:
                gt_snip_2 = embed_in_center(gt_snipped, 96, coords=False)

            masked = image_snip_2[mask_snip_2>0]
            image_snip_2[mask_snip_2>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5)
            image_snip_2[mask_snip_2<1] = 0 

            image_snip_2 = tc.tensor(image_snip_2[None, None, ...], dtype = tc.float32).cuda()
            
            with tc.no_grad():
                output = nn.Sigmoid()(model(image_snip_2))

            if file_gt is not None:
                dice_loss = LOSS(output, tc.tensor(gt_snip_2[None, None, ...], dtype=tc.float32).cuda()) 
                print(f"Teeth number {teeth_num} - Dice loss: {dice_loss.item()}")

            start, end = cords_2
            output = output.detach().cpu().numpy()[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            output[output > THRESHOLD] = 1
            output[output < THRESHOLD] = 0

            z_min, z_max, y_min, y_max, x_min, x_max = coords

            canal_mask_temp = np.zeros(canal_mask.shape)
            canal_mask_temp[z_min:z_max, y_min:y_max, x_min:x_max] = output
            canal_mask[canal_mask_temp == 1] = 1
            del output
            tc.cuda.empty_cache()
        
        else:   
                z_half = image_snipped.shape[0]//2
                image_snip_12, cords_1 = embed_in_center(image_snipped[:z_half], 96, coords=True)
                image_snip_22, cords_2 = embed_in_center(image_snipped[z_half:], 96, coords=True)
                
                mask_snip_12 = embed_in_center(mask_snipped[:z_half], 96, coords=False)
                mask_snip_22 = embed_in_center(mask_snipped[z_half:], 96, coords=False)

                if file_gt is not None:
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
                    pred1 = nn.Sigmoid()(model(image_snip_12))
                
                if file_gt is not None:
                    dice_loss = LOSS(pred1, tc.tensor(gt_snip_12[None, None, ...], dtype=tc.float32).cuda())
                    print(f"Teeth number {teeth_num} - Dice loss: {dice_loss.item()}")
                output[:z_half] = pred1.detach().cpu().numpy()[0, 0, start1[0]:end1[0], start1[1]:end1[1],start1[2]:end1[2]]

                with tc.no_grad():
                    pred2 = nn.Sigmoid()(model(image_snip_22))  
                if file_gt is not None:              
                    dice_loss = LOSS(pred2, tc.tensor(gt_snip_22[None, None, ...], dtype=tc.float32).cuda())
                    print(f"Teeth number {teeth_num} - Dice loss: {dice_loss.item()}")
                output[z_half:] = pred2.detach().cpu().numpy()[0, 0, start2[0]:end2[0], start2[1]:end2[1],start2[2]:end2[2]]
                
                output[output > THRESHOLD] = 1
                output[output < THRESHOLD] = 0

                z_min, z_max, y_min, y_max, x_min, x_max = coords

                canal_mask_temp = np.zeros(canal_mask.shape)
                canal_mask_temp[z_min:z_max, y_min:y_max, x_min:x_max] = output
                canal_mask[canal_mask_temp == 1] = 1

    canal_mask[helper == 0] = 0
    nifti_img = sitk.GetImageFromArray(canal_mask.astype(np.uint8))
    original = sitk.ReadImage(file_instance_seg)
    nifti_img.CopyInformation(original)

    # if DATA == "toothfairy":
    #     sitk.WriteImage(nifti_img, f'../evaluation/evaluation_results/output_canals_mask_tf_{FILE}.nii.gz')
    # else:
    sitk.WriteImage(nifti_img, out)
    
    if file_gt is not None:
        def dice(mask1, mask2):
            tp = np.count_nonzero((mask1+mask2) == 2)
            fp = np.count_nonzero((mask1+mask2) == 1)

            return 2*tp/(2*tp+fp)

        print(f"Dice score: {dice(canal_mask, ground_truth)}")

if __name__ == "__main__":

    PATH1 = '../../weights_tooth_fairy/tooth_fairy_3d_best_agh.pt'
    FILE = 46
    
    for i in range(10):
        
        model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False, weight_path=PATH1)
        model.eval()
        model = model.cuda()
        
        eval(model=model, 
            file_cbct=f'D:/db1_v1.0/db1_v1.0_nifti/{FILE}/image.nii.gz', 
            file_instance_seg = f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{FILE}.nii.gz',
            #file_instance_seg = f'C:/Users/Jan/Downloads/{FILE}.nii.gz',
            file_gt=None, #f'D:/db1_v1.0/db1_v1.0_nifti_root_canal_masks_single_cls_ref/10.nii.gz', 
            out = f'../evaluation/evaluation_results/output_canals_mask_{FILE}_test.nii.gz')
        
        FILE += 1