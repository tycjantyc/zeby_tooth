import numpy as np
import sys
import os
import nibabel as nib
sys.path.append(os.getcwd().removesuffix('\\evaluation'))
print(sys.path)
from monai.transforms import Resize
from models.class_segmentor_model import build_class_segmentor_model
import torch as tc
import torch.nn as nn
from processing.tooth_fairy_classes import teeth_instance, teeth_semantic
from processing.utils import load_itk, crop_to_mask
from evaluation.evaluate_agh_segmentor import evaluate_agh

model = build_class_segmentor_model(inchannels=1, outchannels=34, depths=[32, 64, 128, 256, 384])
model.load_state_dict(tc.load('../models/weights/agh_segmentor_best_1.pt', weights_only=True))
#file = 'D:/ToothFairy3/labelsTr/ToothFairy3F_001.nii'
file = 'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/4.nii.gz'
file2 = 'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/4.nii.gz'

image = load_itk(file)
mask = load_itk(file2)

image, mask = np.flip(image, axis=(0, 2)), np.flip(mask, axis=(0, 2))

# image = teeth_semantic(image)
# mask = teeth_instance(mask)

image, mask = crop_to_mask(image, mask, padding=2)

label_weak = image.astype(np.float32)
label = mask.astype(np.float32)

print(label_weak.shape)

down = Resize(spatial_size = (128, 128, 128), 
                      size_mode='all', 
                      mode="trilinear", 
                      align_corners=None, 
                      anti_aliasing=True, 
                      anti_aliasing_sigma=None,
                      dtype=np.float32, 
                      lazy=False)
        
label_weak = down.__call__(label_weak[None, ...])
label = down.__call__(label[None, ...])[0]

model = model.cuda()

image = tc.tensor(label_weak[None, ...], dtype=tc.float32).cuda()

output = nn.Softmax(dim=1)(model(image))

output = tc.argmax(output, dim=1)

output = output[0].detach().cpu().numpy().astype(np.uint8)

print(output.shape)

affine = np.eye(4)
nifti_img = nib.Nifti1Image(output, affine)
nib.save(nifti_img, '../evaluation/evaluation_results/output_image_1.nii.gz')


