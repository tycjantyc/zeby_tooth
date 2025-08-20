import numpy as np
import SimpleITK as sitk

def load_itk(filename:str, clamp:tuple[int, int] = None):
    """
    For teeth and canals often clamp is (500, 3000)
    """
    itkimage = sitk.ReadImage(filename)
    if clamp is not None:
        itkimage = sitk.Clamp(itkimage, lowerBound=clamp[0], upperBound=clamp[1])

    ct_scan = sitk.GetArrayFromImage(itkimage)
    return ct_scan

def norm_to_0_1(image: np.ndarray, min = -1000, max = 3000):
    return (image - min)/(max - min)

def norm_0_1_min_max(image: np.ndarray):
    return (image - image.min())/(image.max() - image.min())

def norm_factory(mode_name = 'none'):
    if mode_name == 'none':
        return lambda x:x
    
    elif mode_name == 'minmax':
        return norm_0_1_min_max
    
    elif mode_name == '01':
        return norm_to_0_1
    
    else:
        raise ValueError('There is no such mode!')


def crop(image: np.ndarray, new_dims: list[int, int, int]):

    h, w, d = image.shape
    h1, w1, d1 = new_dims

    x1 = int((h-h1)/2)
    y1 = h - x1

    x2 = int((w-w1)/2)
    y2 = w - x2

    x3 = int((d-d1)/2)
    y3 = d - x3

    return image[x1:y1, x2:y2, x3:y3]


def crop_to_mask(volume:  np.ndarray, mask: np.ndarray, padding=2, input = 96, coordss = False):

    coords = np.array(np.nonzero(mask))
    if coords.sum() == 0:
         return volume, mask

    zmin, ymin, xmin = coords.min(axis=1)
    zmax, ymax, xmax = coords.max(axis=1)

    zmin = max(zmin - padding, 0)
    ymin = max(ymin - padding, 0)
    xmin = max(xmin - padding, 0)

    zmax = min(zmax + padding, volume.shape[0])
    ymax = min(ymax + padding, volume.shape[1])
    xmax = min(xmax + padding, volume.shape[2])

    cropped_volume = volume[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    if coordss:
        return cropped_volume, cropped_mask, (zmin, zmax, ymin, ymax, xmin, xmax)
    else:
        return cropped_volume, cropped_mask


def create_random_snippet(image: np.ndarray, mask:np.ndarray, input_size: tuple[int, int, int], coords = False):
        h, w, d = image.shape
        h1, w1, d1 = input_size

        if h < h1 or w < w1 or d < d1:
            raise ValueError('Input size is too big!')
        
        num = 0

        while True:
            h_new = np.random.randint(0, h-h1)
            w_new = np.random.randint(0, w-w1)
            d_new = np.random.randint(0, d-d1)

            temp_mask = mask[h_new:h_new+h1, w_new:w_new+w1, d_new:d_new+d1]

            if np.all(mask == 0) and num < 20:   #prevents from choosing meaningless masks
                num += 1
                continue

            temp_image = image[h_new:h_new+h1, w_new:w_new+w1, d_new:d_new+d1]

            coordianates = [[h_new/h, (h_new+h1)/h], [w_new/w, (w_new+h1)/w], [d_new/d, (d_new+d1)/d]]

            if not coords:
                return temp_image, temp_mask
            
            else:
                return temp_image, temp_mask, coordianates
            

import numpy as np

def embed_in_center(small_volume: np.ndarray, big_volume_size: int, coords:bool = False) -> np.ndarray:
    
    small_shape = small_volume.shape    
    big_volume = np.zeros((big_volume_size, big_volume_size, big_volume_size), dtype=small_volume.dtype)

    start = [(big_volume_size - s) // 2 for s in small_shape]
    end = [start[i] + small_shape[i] for i in range(3)]

    big_volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = small_volume

    if coords:
        return big_volume, (start, end)
    else:
        return big_volume
    
import numpy as np
from scipy.ndimage import (
    generate_binary_structure,
    binary_closing,
    binary_opening,
    binary_fill_holes,
)

import numpy as np
from scipy import ndimage as ndi

def infill_label(
    labels: np.ndarray,
    target: int,
    closing_iters: int = 4,
    method: str = "closing",
    background: int = 0,
):
    """
    Fill every cavity that belongs to `target` inside a multi-label mask.

    Parameters
    ----------
    labels : ndarray
        Shape (Z, Y, X). Each voxel contains an integer class index.
    target : int
        The class to make solid.
    closing_iters : int, default 2
        Number of morphological-closing iterations (ignored if `method='flood'`).
    method : {'closing', 'flood'}, default 'closing'
        * 'closing'  – fast, seals small cracks first, then fills holes.
        * 'flood'    – guaranteed watertight even with huge gaps, but slower.
    background : int, default 0
        The value that represents background / unlabeled voxels.

    Returns
    -------
    solid : ndarray
        Same shape & dtype as `labels`, but `target` is now solid.
    """
    solid = labels.copy()
    mask = (labels == target)                      # binary view of the class
    struct = ndi.generate_binary_structure(3, 1)   # 6-connected neighbourhood
    struct = np.ones((3, 3, 3)) 

    if method == "closing":
        # 1) seal leaks • 2) fill enclosed cavities
        closed  = ndi.binary_closing(mask, structure=struct,
                                     iterations=closing_iters)
        filled  = ndi.binary_fill_holes(closed)

    if method == "infill":
            # 1) seal leaks • 2) fill enclosed cavities
        
            filled  = ndi.binary_fill_holes(mask*1)
    
    elif method == "flood":
        # “Exterior flood-fill” → always watertight
        mask = np.pad(mask, 4, constant_values=0)
        mask  = ndi.binary_closing(mask, structure=struct,
                                     iterations=3)*1
        mask = mask[4:-4, 4:-4, 4:-4]
        padded  = np.pad(mask, 1, constant_values=1)
        exterior = ndi.binary_fill_holes(padded[:, 1:-1, 1:-1])
        filled   = exterior[1:-1]
    else:
        raise ValueError("method must be 'closing' or 'flood'")

    # Add only the *new* interior voxels and only where background now
    new_voxels = filled & (solid == background)
    solid[new_voxels] = target
    return solid
