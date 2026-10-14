"""AGH sliding-window inference with intensity and coordinate channels."""

import numpy as np
from torch import nn

from inference.volume import predict_volume
from processing.utils import load_itk, norm_to_0_1
from processing.prepare_tooth_fairy_data import prepare_toothfairy


def evaluate_agh(model: nn.Module, file: str, model_input_shape: tuple[int, int, int],
                 stride: tuple[int, int, int], num_classes: int = 33,
                 remove_ct_rings: bool = False,
                 normalization_function=norm_to_0_1, channels=None, device="cuda"):
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
        augmentation=False, channels=channels or [], patchify=False, coords=False)
    scores = predict_volume(model, volume, model_input_shape, stride,
                            num_classes, device=device, coordinates=True)
    if num_classes == 1:
        return (scores[0] >= 0.5).astype(np.uint8)
    return scores.argmax(axis=0).astype(np.uint8 if num_classes <= 256 else np.uint16)

