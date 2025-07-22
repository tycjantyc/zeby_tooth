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
             num_classes:int = 3,  
             remove_ct_rings:bool = False, 
             normalization_function = norm_to_0_1, 
             channels:list = []):
    
    volume = load_itk(file)
    volume, _ = prepare_toothfairy(volume, volume.copy(), remove_ct_rings=remove_ct_rings, input_size=volume.shape, normalization_function=normalization_function, augmentation=False, channels=channels, patchify=False, coords=False)
    volume = volume[0]

    model = model.cuda()
    
    h, w, d = model_input_shape

    patches, indices, padded_shape = get_patches(volume, model_input_shape, stride)

    output = np.zeros((num_classes, *padded_shape)) 
    counts = np.zeros_like(output)

    for patch, (i, j, k) in tqdm(zip(patches, indices)):
        
        x, y, z = i * stride[0], j * stride[1], k * stride[2]

        coords = ((x+h/2)/volume.shape[0], (y+w/2)/volume.shape[1], (z+d/2)/volume.shape[2])
        
        x = np.ones((1, *volume.shape), dtype = np.float32) * coords[0]
        y = np.ones((1, *volume.shape), dtype = np.float32) * coords[1]
        z = np.ones((1, *volume.shape), dtype = np.float32) * coords[2]

        image = np.concatenate((image, x, y, z), axis = 0)
        
        input_patch = patch[np.newaxis, ...]  # Adjust to your model's input shape
        input_patch = tc.tensor(input_patch, dtype = tc.float32).cuda()
        prediction = model(input_patch)  
        prediction = nn.Sigmoid()(prediction)
        prediction = prediction.squeeze().detach().cpu().numpy()  
        
        output[:, x:x+h, y:y+w, z:z+d] += prediction
        counts[:, x:x+h, y:y+w, z:z+d] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    segmented_volume = output / counts

    #Crop to original
    start = []
    end = []
    for i in range(3):
        start.append((padded_shape[i] - volume.shape[i]) // 2)
        end.append(start[i] + volume.shape[i])
    segmented_volume = segmented_volume[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    print(segmented_volume.shape)   
    segmented_volume = np.argmax(segmented_volume, axis=0)
    print(segmented_volume.shape)

    return segmented_volume.astype(np.uint8)

