#!/usr/bin/env python3
import argparse
import numpy as np
import SimpleITK as sitk
import torch
from torch import tensor
import torch.nn.functional as F
import torch
from processing.utils import infill_label

def nearest_active_voxel(mask: np.ndarray, coord):
    
    if mask.ndim != 3:
        raise ValueError("mask must be 3D (D, H, W)")

    coord = np.array(coord, dtype=float)

    # Get all coordinates where mask == 1
    active = np.argwhere(mask == 1)
    if active.size == 0:
        return None  # no 1s in mask

    # Compute squared Euclidean distances to all active voxels
    diffs = active - coord
    dists = np.sum(diffs ** 2, axis=1)

    # Find index of the smallest distance
    nearest_idx = np.argmin(dists)
    nearest = active[nearest_idx]

    return nearest

import torch
import numpy as np
from skimage.measure import label, regionprops

def label_and_bbox_centers(vol: torch.Tensor, connectivity: int = 1):
    
    if vol.ndim != 3:
        raise ValueError("Input must be 3D (D, H, W)")

    vol_np = vol.cpu().numpy().astype(bool)

    labeled_np, _ = label(vol_np)

    props = regionprops(labeled_np)
    centers = []
    for p in props:
        zmin, ymin, xmin, zmax, ymax, xmax = p.bbox
        centers.append([(zmin + zmax - 1) / 2.0,
                        (ymin + ymax - 1) / 2.0,
                        (xmin + xmax - 1) / 2.0])

    centers = torch.tensor(centers, dtype=torch.float32, device=vol.device)
    labels = torch.from_numpy(labeled_np).to(vol.device)

    return centers



import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, label
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.segmentation import random_walker

def watershed_from_centroids(
    semantic_mask_path: str,
    centroids: np.ndarray
) -> np.ndarray:
    
    try:
        sitk_image = sitk.ReadImage(semantic_mask_path)
        semantic_mask = sitk.GetArrayFromImage(sitk_image)
        semantic_mask = infill_label(semantic_mask, target=1, method='flood')
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return None

    # Upewnij się, że maska jest binarna (0 dla tła, 1 dla zębów)
    binary_mask = (semantic_mask > 0).astype(np.uint8)

    # --- Krok 2: Przygotuj markery na podstawie centroidów ---
    # Stwórz pustą mapę o wymiarach maski
    markers = np.zeros(binary_mask.shape, dtype=np.int32)

    temp = binary_mask.copy()
    # Umieść każdy centroid na mapie jako unikalny punkt
    for i, centroid in enumerate(centroids):
        # Współrzędne muszą być liczbami całkowitymi do indeksowania
        z, y, x = int(centroid[0]), int(centroid[1]), int(centroid[2])

        # Upewnij się, że współrzędne mieszczą się w granicach obrazu
        if 0 <= z < markers.shape[0] and 0 <= y < markers.shape[1] and 0 <= x < markers.shape[2]:
            # Przypisz unikalną, dodatnią etykietę dla każdego centroidu
            
            if binary_mask[z, y, x] > 0:
                pass
            else:
                z, y, x = nearest_active_voxel(binary_mask, np.array([z, y, x]))
            
            temp[z-3:z+3, y-3:y+3, x-3:x+3] = 2
            markers[z, y, x] = i + 1  

    temp_image = sitk.GetImageFromArray(temp)
    sitk.WriteImage(temp_image, f'C:/Users/Jan/Desktop/tooth_fairy_challenge/evaluation/evaluation_results/watershed_helper_30.nii.gz')
    print("Saved helper!")
    

    # --- Krok 4: Uruchom algorytm watershed ---
    print("Starting random walker...")
    
    instance_mask = random_walker(binary_mask, markers, beta=10, mode='cg_mg')
    print("Ended random walker with success!")

    return instance_mask


if __name__ == '__main__':

    NUMBER = 30

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-mask", required=False, default=f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/{NUMBER}.nii.gz')
    ap.add_argument("--heat-map", required=False, default=f'C:/Users/Jan/Desktop/tooth_fairy_challenge/evaluation/evaluation_results/heatmap_{NUMBER}.nii.gz')
    ap.add_argument("--out", required=False, default=f'C:/Users/Jan/Desktop/tooth_fairy_challenge/evaluation/evaluation_results/watershed_{NUMBER}_2.nii.gz')
    args = ap.parse_args()

    image = sitk.ReadImage(args.heat_map)
    heatmap = sitk.GetArrayFromImage(image)
    heatmap = tensor([heatmap], dtype=torch.float32)
    print(f"Max heatmap:{heatmap.max()}")
    # heatmap[heatmap<127] = 0
    # heatmap[heatmap>=127] = 1

    print("Calculating centroids coordinates...")
    centroids = label_and_bbox_centers(heatmap[0])
    print(centroids.shape)
    instance_segm = watershed_from_centroids(semantic_mask_path=args.seed_mask, centroids=centroids)

    image_instance = sitk.GetImageFromArray(instance_segm)

    # Save output
    sitk.WriteImage(image_instance, args.out)
    print(f"Saved watershed labels to: {args.out}")