"""Sliding-window segmentation, returning a NumPy label volume."""

import numpy as np
from torch import nn

from inference.volume import predict_volume
from processing.utils import load_itk, norm_to_0_1
from processing.tooth_fairy_classes import only_teeth_and_canal_classes
from processing.prepare_tooth_fairy_data import prepare_toothfairy


def evaluate(model: nn.Module, file: str, model_input_shape: tuple[int, int, int],
             stride: tuple[int, int, int], num_classes: int = 3,
             return_binary: bool = False, remove_ct_rings: bool = False,
             compression_function=only_teeth_and_canal_classes,
             normalization_function=norm_to_0_1, channels=None, device="cuda"):
    # compression_function is retained for compatibility; labels are determined
    # by the checkpoint. Compression was never applied during inference.
    file = str(file)
    if file.lower().endswith((".nii", ".nii.gz")):
        volume = load_itk(file)
    elif file.lower().endswith(".npy"):
        volume = np.load(file)
    else:
        raise ValueError("Expected .nii, .nii.gz or .npy")
    volume, _ = prepare_toothfairy(
        volume, volume.copy(), remove_ct_rings=remove_ct_rings,
        input_size=volume.shape, normalization_function=normalization_function,
        augmentation=False, channels=channels or [], patchify=False)
    scores = predict_volume(model, volume, model_input_shape, stride,
                            num_classes, device=device)
    if return_binary:
        return (scores >= 0.5).astype(np.uint8)
    if num_classes == 1:
        return (scores[0] >= 0.5).astype(np.uint8)
    return scores.argmax(axis=0).astype(np.uint8 if num_classes <= 256 else np.uint16)

