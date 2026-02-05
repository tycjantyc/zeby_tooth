import numpy as np
import sys
import os
import nibabel as nib
import SimpleITK as sitk
sys.path.append(os.getcwd().removesuffix('\\evaluation'))
print(sys.path)
from monai.transforms import Resize
from models.class_segmentor_model import build_class_segmentor_model
import torch as tc
import torch.nn as nn
from processing.tooth_fairy_classes import teeth_instance, teeth_semantic
from processing.utils import load_itk, crop_to_mask
from evaluation.evaluate_agh_segmentor import evaluate_agh
from monai.losses import DiceLoss
import torch.nn.functional as F

def instance_teeth_segment(model, file_semantic, file_instance, out):
    LOSS = DiceLoss(include_background=False, reduction = 'none') 
    
    image = load_itk(file_semantic)
    if file_instance is not None:
        mask = load_itk(file_instance)

    image = np.flip(image, axis=(0, 2))

    if file_instance is not None:
        mask = np.flip(mask, axis=(0, 2))

    # image = teeth_semantic(image)
    # mask = teeth_instance(mask)
    if file_instance is not None:
        image, mask, coords = crop_to_mask(image, mask, padding=2, coordss=True)
    else:
        image, _, coords = crop_to_mask(image, image, padding=2, coordss=True)

    label_weak = image.astype(np.float32)

    if file_instance is not None:
        label = mask.astype(np.float32)

    print(label_weak.shape)

    down = Resize(spatial_size = (128, 128, 128), 
                        size_mode='all', 
                        mode="nearest-exact", 
                        align_corners=None, 
                        anti_aliasing=False, 
                        anti_aliasing_sigma=None,
                        dtype=np.float32, 
                        lazy=False)

    up = Resize(spatial_size = image.shape, 
                        size_mode='all', 
                        mode="nearest", 
                        align_corners=None, 
                        anti_aliasing=False, 
                        anti_aliasing_sigma=None,
                        dtype=np.uint8, 
                        lazy=False)
            
    label_weak = down.__call__(label_weak[None, ...])
    if file_instance is not None:
        label = down.__call__(label[None, ...])
    model = model.cuda()

    image = tc.tensor(label_weak[None, ...], dtype=tc.float32).cuda()
    if file_instance is not None:
        label = tc.tensor(label, dtype=tc.float32).cuda()

    output = nn.Softmax(dim=1)(model(image))


    if file_instance is not None:
        clean_output = F.one_hot(label.long(), num_classes=34)
        clean_output = clean_output.permute(0, 4, 1, 2, 3).float()

        print(LOSS(output, clean_output))
        print(f"Dice score: {1 - LOSS(output, clean_output).item()}")

    output = tc.argmax(output, dim=1)
    output = output[0].detach().cpu().numpy().astype(np.uint8)
    print(output.shape)

    output = up.__call__(output[None, ...])[0]
    temp = np.zeros(load_itk(file_semantic).shape)
    z_min, z_max, y_min, y_max, x_min, x_max = coords
    print(output.dtype)
    temp[z_min:z_max, y_min:y_max, x_min:x_max] = output
    output = temp

    affine = np.eye(4)
    original = sitk.ReadImage(file_semantic)
    nifti_img = sitk.GetImageFromArray(np.flip(output, axis = (0, 2)))
    nifti_img.CopyInformation(original)
    sitk.WriteImage(nifti_img, out)
    # nifti_img2 = nib.Nifti1Image(label[0].detach().cpu().numpy().astype(np.uint8), affine)
    # nib.save(nifti_img2, '../evaluation/evaluation_results/output_image_2_mask.nii.gz')
    # nifti_img3 = nib.Nifti1Image(label_weak[0].numpy().astype(np.uint8), affine)
    # nib.save(nifti_img3, '../evaluation/evaluation_results/output_image_2_01.nii.gz')
    print("DONE!")

if __name__ == "__main__":
    model = build_class_segmentor_model(inchannels=1, outchannels=34, depths=[32, 64, 128, 256, 384])
    model.load_state_dict(tc.load('../models/weights/agh_segmentor_best_2.pt', weights_only=True)) #best_2 są ok!
    #model.load_state_dict(tc.load("C:/Users/Jan/Desktop/weights_tooth_fairy/agh_segmentor_best.pt", weights_only=True))

    file_semantic = 'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/25.nii.gz'
    file_instance = None #'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/17.nii.gz'
    out = '../evaluation/evaluation_results/prediction_sepration_25.nii.gz'

    instance_teeth_segment(model, file_semantic, file_instance, out)

