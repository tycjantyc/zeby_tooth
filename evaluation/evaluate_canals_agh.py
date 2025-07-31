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
from evaluation.evaluate_agh_segmentor import evaluate_agh
from processing.utils import load_itk, crop_to_mask
from processing.utils import embed_in_center
from tqdm import tqdm

model =build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False)
model.load_state_dict(tc.load('../models/weights/tooth_fairy_3d_best_agh.pt', weights_only=True))
model.eval()

file = 'D:/db1_v1.0/db1_v1.0_nifti/1/image.nii.gz'
helper_path = 'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/1.nii.gz'

model = model.cuda()

image = load_itk(file)
helper = load_itk(helper_path)

print(image.shape)

canal_mask = np.zeros(image.shape)

for teeth_num in tqdm(np.unique(helper)[1:]):

    temp_mask = np.zeros(helper.shape)

    temp_mask[helper != teeth_num] = 0
    temp_mask[helper == teeth_num] = 1 

    image_snipped, mask_snipped, coords = crop_to_mask(image, temp_mask, padding=2, input=1, coordss=True)
    
    print(image_snipped.shape)

    if ((image_snipped.shape[0] < 96) and (image_snipped.shape[1] < 96) and (image_snipped.shape[2] < 96)): 
        print('ok')
        image_snip_2, cords_2 = embed_in_center(image_snipped, 96, coords=True)
        mask_snip_2 = embed_in_center(image_snipped, 96, coords=False)

        masked = image_snip_2[mask_snip_2>0]
        image_snip_2[mask_snip_2>0] = (masked - masked.min())/(masked.max() - masked.min() + 1e-5)
        image_snip_2[mask_snip_2<1] = 0 


        image_snip_2 = tc.tensor(image_snip_2[None, None, ...], dtype = tc.float32).cuda()

        output = nn.Sigmoid()(model(image_snip_2)).detach().cpu().numpy()[0, 0]
        output[output > 0.5] = 1
        output[output < 0.5] = 0

        z_min, z_max, y_min, y_max, x_min, x_max = coords
        start, end = cords_2
        print(coords)
        print(cords_2)

        z1 = max(0, z_min-start[0])

        canal_mask[z1:z_max+96-end[0], y_min - start[1]:y_max+96-end[1], x_min - start[2]:x_max+96-end[2]] = output[z1 - z_min + start[0]:]
     
    else:   
            z_half = image_snipped.shape[0]//2
            image_snip_12, cords_1 = embed_in_center(image_snipped[:z_half], 96, coords=True)
            image_snip_22, cords_2 = embed_in_center(image_snipped[z_half:], 96, coords=True)
            
            mask_snip_12 = embed_in_center(mask_snipped[:z_half], 96, coords=False)
            mask_snip_22 = embed_in_center(mask_snipped[z_half:], 96, coords=False)

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

            print(coords)
            print(cords_1)
            print(cords_2)

            output[:z_half] = nn.Sigmoid()(model(image_snip_12)).detach().cpu().numpy()[0, 0, start1[0]:end1[0], start1[1]:end1[1],start1[2]:end1[2]]
            output[z_half:] = nn.Sigmoid()(model(image_snip_22)).detach().cpu().numpy()[0, 0, start2[0]:end2[0], start2[1]:end2[1],start2[2]:end2[2]]
            
            output[output > 0.5] = 1
            output[output < 0.5] = 0

            z_min, z_max, y_min, y_max, x_min, x_max = coords
        
            canal_mask[z_min:z_max, y_min:y_max, x_min:x_max] = output


canal_mask[helper == 0] = 0
nifti_img = sitk.GetImageFromArray(canal_mask.astype(np.uint8))
original = sitk.ReadImage(helper_path)
nifti_img.CopyInformation(original)

sitk.WriteImage(nifti_img, '../evaluation/evaluation_results/output_canals_mask_2.nii.gz')