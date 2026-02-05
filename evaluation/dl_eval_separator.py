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
from evaluation.test_centroid import instance_teeth_segment
from processing.utils import crop_to_mask_v2
from tqdm import tqdm

def dl_teeth_segment(model, model2, file_semantic, end):
    
    image = load_itk(file_semantic)
    output = np.zeros_like(image)

    output = np.pad(output, 64)
    teeth_mask_temp = np.zeros_like(output)

    centroids = instance_teeth_segment(model, file_semantic, return_file=False)
    centroids = np.flip(centroids, axis = (0, 2))

    model2 = model2.cuda()

    for i in tqdm(np.unique(centroids)[1:]):
        
        image_temp, _, coords = crop_to_mask_v2(image, (centroids==i)*1, return_coords=True)
        image_temp = tc.tensor(image_temp[None, None, :, :, :], dtype = tc.float32)

        image_temp = image_temp.cuda()

        out = nn.Softmax(dim = 1)(model2(image_temp))
        out = tc.argmax(out, dim = 1)

        out = out.cpu().detach().numpy()[0]

        out[out == 1] = 0
        out[out == 2] = i

        zmin, ymin, xmin, zmax, ymax, xmax, slide = coords
        z = (zmax+zmin)//2
        y = (ymax+ymin)//2 
        x = (xmax+xmin)//2

        teeth_mask_temp[z-slide:z+slide, y-slide:y+slide, x-slide:x+slide] = out   
        output[teeth_mask_temp == i] = i

    output = output[64:-64, 64:-64, 64:-64]

    original = sitk.ReadImage(file_semantic)
    nifti_img = sitk.GetImageFromArray(output) #np.flip(output, axis = (0, 2))
    nifti_img.CopyInformation(original)
    sitk.WriteImage(nifti_img, end)
    print("DONE!")

    
if __name__ == "__main__":
    model = build_class_segmentor_model(inchannels=1, outchannels=1, depths=[32, 64, 128, 256, 384])
    model.load_state_dict(tc.load('../weights_tooth_fairy/agh_centroid_best_s2.pt', weights_only=True)) 

    model2 = build_class_segmentor_model(inchannels=1, outchannels=3, depths=[32, 64, 128, 256, 384])
    model2.load_state_dict(tc.load('models/weights/dl_separator_best_v2.pt', weights_only=True)) 
    for i in range(3, 4):
        file_semantic = f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/{i}.nii.gz'
        out = f'evaluation/evaluation_results/dl_separation_{i}_new.nii.gz'
        
        dl_teeth_segment(model, model2, file_semantic, out)