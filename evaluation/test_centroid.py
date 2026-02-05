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
from processing.utils import load_itk, crop_to_mask
from skimage.measure import label
import torch.nn.functional as F


def instance_teeth_segment(model, file_semantic, out = 'ok', return_file = True):
    
    if type(file_semantic) == str:

        image = load_itk(file_semantic)
    
    else:
        image = file_semantic

    image = np.flip(image, axis=(0, 2))

    temp = np.zeros_like(image)
  
    image, _, coords = crop_to_mask(image, image, padding=2, coordss=True)

    label_weak = image.astype(np.float32)

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
 
    model = model.cuda()

    image = tc.tensor(label_weak[None, ...], dtype=tc.float32).cuda()
   
    output = nn.Sigmoid()(model(image))
    #output = output*255
    output[output <= 0.5] = 0
    output[output > 0.5] = 1

    output = output[0, 0].detach().cpu().numpy().astype(np.uint16)

    output = up.__call__(output[None, ...])[0]
    
    z_min, z_max, y_min, y_max, x_min, x_max = coords
    temp[z_min:z_max, y_min:y_max, x_min:x_max] = output
    output = temp

    output = label(output)

    if return_file:

        original = sitk.ReadImage(file_semantic)
        nifti_img = sitk.GetImageFromArray(np.flip(output, axis = (0, 2)))
        nifti_img.CopyInformation(original)
        sitk.WriteImage(nifti_img, out)
        print("DONE!")

    else:
        return output

if __name__ == "__main__":
    model = build_class_segmentor_model(inchannels=1, outchannels=1, depths=[32, 64, 128, 256, 384])
    model.load_state_dict(tc.load('../weights_tooth_fairy/agh_centroid_best_dice+centerloss.pt', weights_only=True))  #agh_centroid_best_s2 #agh_centroid_best_new_loss_4

    file_semantic = 'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/101.nii.gz'
    out = 'evaluation/evaluation_results/heatmap_101_new_2.nii.gz'

    instance_teeth_segment(model, file_semantic, out)

