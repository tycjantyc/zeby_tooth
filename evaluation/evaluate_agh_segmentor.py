from torch import nn
import numpy as np
from skimage.util import view_as_windows
from processing.utils import load_itk, norm_to_0_1
from processing.tooth_fairy_classes import only_teeth_and_canal_classes
import torch as tc
from tqdm import tqdm
from processing.prepare_tooth_fairy_data import prepare_toothfairy

patch_size = (96, 96, 96)
stride = (48, 48, 48)  

def get_patches(volume, patch_size, stride):
    padded = np.pad(volume, [(max(0, (patch_size[i] - volume.shape[i] % patch_size[i])//2), max(0, (patch_size[i] - volume.shape[i] % patch_size[i]))//2) for i in range(3)])
    windows = view_as_windows(padded, patch_size, step=stride)
    indices = [(i, j, k) for i in range(windows.shape[0])
                         for j in range(windows.shape[1])
                         for k in range(windows.shape[2])]
    patches = [windows[i, j, k] for i, j, k in indices]
    return patches, indices, padded.shape


def evaluate_agh(model:nn.Module, 
             file:str, 
             model_input_shape: tuple[int, int, int], 
             stride: tuple[int, int, int], 
             num_classes:int = 33,  
             remove_ct_rings:bool = False, 
             normalization_function = norm_to_0_1, 
             channels:list = []):
    
    if file.endswith('nii.gz'):
        volume = load_itk(file)
    elif file.endswith('.npy'):
        volume = np.load(file)
        
    volume, _ = prepare_toothfairy(volume, volume.copy(), remove_ct_rings=remove_ct_rings, input_size=volume.shape, normalization_function=normalization_function, augmentation=False, channels=channels, patchify=False, coords=False)
    volume = volume[0]

    model = model.cuda()
    
    h, w, d = model_input_shape

    patches, indices, padded_shape = get_patches(volume, model_input_shape, stride)

    output = np.zeros((num_classes, *padded_shape), dtype = np.float16) 
    counts = np.zeros_like(output, dtype=np.uint8)

    for patch, (i, j, k) in tqdm(zip(patches, indices)):
        
        x, y, z = i * stride[0], j * stride[1], k * stride[2]

        coords = ((x+h/2)/volume.shape[0], (y+w/2)/volume.shape[1], (z+d/2)/volume.shape[2])
        
        x1 = np.ones((1, *patch.shape), dtype = np.float32) * coords[0]
        y1 = np.ones((1, *patch.shape), dtype = np.float32) * coords[1]
        z1 = np.ones((1, *patch.shape), dtype = np.float32) * coords[2]

        patch = np.concatenate((patch[None, ...], x1, y1, z1), axis = 0)
        
        input_patch = patch[np.newaxis, ...]  # Adjust to your model's input shape
        input_patch = tc.tensor(input_patch, dtype = tc.float32).cuda()
        prediction = model(input_patch)  
        prediction = nn.Sigmoid()(prediction)
        prediction = prediction.squeeze().detach().cpu().numpy()  
        
        output[:, x:x+h, y:y+w, z:z+d] += prediction.astype(np.float16)
        counts[:, x:x+h, y:y+w, z:z+d] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    segmented_volume = output / counts

    del output, counts

    #Crop to original
    start = []
    end = []
    for i in range(3):
        start.append((padded_shape[i] - volume.shape[i]) // 2)
        end.append(start[i] + volume.shape[i])
    segmented_volume = segmented_volume[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    print(segmented_volume.shape)   
    
    labels    = np.zeros(volume.shape, dtype=np.uint8)   
    best_vals = segmented_volume[0].copy()            
    for c in range(1, 33):
        cls_vals = segmented_volume[c]               
        mask = cls_vals > best_vals
        labels[mask]    = c
        best_vals[mask] = cls_vals[mask]
    del best_vals, segmented_volume                   
    return labels

    #segmented_volume = np.argmax(segmented_volume, axis=0)
    # print(segmented_volume.shape)
    # return segmented_volume.astype(np.uint8)

