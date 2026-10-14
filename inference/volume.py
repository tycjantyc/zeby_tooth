"""Sliding-window inference and geometry-preserving NIfTI IO."""

from itertools import product
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch


def read_volume(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not str(path).lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Expected a .nii or .nii.gz file")
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D volume")
    array = sitk.GetArrayFromImage(image)  # Z, Y, X throughout the pipeline
    if not np.isfinite(array).all():
        raise ValueError("Input contains NaN or infinity")
    return image, array


def require_same_geometry(reference, mask):
    if reference.GetSize() != mask.GetSize() or any(
        not np.allclose(a, b, rtol=0, atol=1e-5)
        for a, b in (
            (reference.GetSpacing(), mask.GetSpacing()),
            (reference.GetOrigin(), mask.GetOrigin()),
            (reference.GetDirection(), mask.GetDirection()),
        )
    ):
        raise ValueError("CBCT and mask must have the same size, spacing, origin and direction")


def save_mask(array, reference, path):
    if tuple(array.shape) != tuple(reversed(reference.GetSize())):
        raise ValueError("Output mask shape differs from the reference image")
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path), True)
    return str(Path(path).resolve())


def _starts(length, patch, stride):
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.inference_mode()
def predict_volume(model, volume, patch_size, stride, num_classes, *,
                   device="cuda", coordinates=False):
    """Average sigmoid probabilities; cover all voxels, including the last window.

    volume has shape (channels, Z, Y, X). Coordinate channels, when requested,
    use the window centre relative to the original volume, as in AGH training.
    The CPU accumulator uses float32 to avoid overlap overflow.
    """
    if (volume.ndim != 4 or len(patch_size) != 3 or len(stride) != 3
            or num_classes < 1 or any(s < 1 or p < 1 or s > p
                                     for p, s in zip(patch_size, stride))):
        raise ValueError("Expected CZYX volume and 0 < stride <= patch size")
    shape = volume.shape[1:]
    padding = [(max(0, p - n) // 2, max(0, p - n) - max(0, p - n) // 2)
               for n, p in zip(shape, patch_size)]
    padded = np.pad(volume, [(0, 0), *padding])
    scores = np.zeros((num_classes, *padded.shape[1:]), dtype=np.float32)
    counts = np.zeros(padded.shape[1:], dtype=np.uint32)
    model.to(device).eval()
    for start in product(*[_starts(n, p, s) for n, p, s
                            in zip(padded.shape[1:], patch_size, stride)]):
        region = tuple(slice(v, v + p) for v, p in zip(start, patch_size))
        patch = padded[(slice(None), *region)]
        if coordinates:
            centre = [(v - pad[0] + p / 2) / n
                      for v, pad, p, n in zip(start, padding, patch_size, shape)]
            coords = np.stack([np.full(patch_size, c, dtype=np.float32) for c in centre])
            patch = np.concatenate((patch, coords))
        tensor = torch.from_numpy(np.ascontiguousarray(patch[None], dtype=np.float32)).to(device)
        prediction = model(tensor)
        if tuple(prediction.shape) != (1, num_classes, *patch_size):
            raise ValueError(f"Unexpected model output shape: {tuple(prediction.shape)}")
        prediction = prediction.sigmoid().float().cpu().numpy()[0]
        scores[(slice(None), *region)] += prediction
        counts[region] += 1
    scores /= counts[None]
    crop = tuple(slice(p[0], p[0] + n) for p, n in zip(padding, shape))
    return scores[(slice(None), *crop)]
