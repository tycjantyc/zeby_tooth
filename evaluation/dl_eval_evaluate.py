import numpy as np
from processing.utils import load_itk
from tqdm import tqdm
import SimpleITK as sitk

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as e:
    raise ImportError("scipy is required: pip install scipy") from e


def dice_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.count_nonzero(a & b)
    na = np.count_nonzero(a)
    nb = np.count_nonzero(b)
    if na + nb == 0:
        return 1.0
    return 2.0 * inter / (na + nb)


def mean_dice_teeth_hungarian(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    background: int = 0,
    min_dice: float = 0.0,
) -> float:
    """
    Compute mean Dice over teeth after optimal label matching (Hungarian).

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        Same-shape labeled 3D volumes. Teeth labels are integers, background is `background`.
    background : int
        Label to ignore (usually 0).
    min_dice : float
        Disallow matches with Dice < min_dice (treated as impossible).

    Returns
    -------
    float
        Mean Dice across matched teeth. Returns np.nan if no valid matches.
    """
    #mask_b = np.flip(mask_b, axis=(0, 2))

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")

    labels_a = sorted(int(x) for x in np.unique(mask_a) if int(x) != background)
    labels_b = sorted(int(x) for x in np.unique(mask_b) if int(x) != background)

    if not labels_a or not labels_b:
        return float("nan")

    # Precompute boolean masks per label (fast enough for ~33 labels)
    bool_a = {la: (mask_a == la) for la in labels_a}
    bool_b = {lb: (mask_b == lb) for lb in labels_b}

    dice_mat = np.zeros((len(labels_a), len(labels_b)), dtype=float)
    for i, la in tqdm(enumerate(labels_a)):
        for j, lb in enumerate(labels_b):
            dice_mat[i, j] = dice_binary(bool_a[la], bool_b[lb])

    # Maximize Dice => minimize (1 - Dice)
    cost = 1.0 - dice_mat

    # Optionally disallow weak matches
    if min_dice > 0.0:
        cost = cost.copy()
        cost[dice_mat < min_dice] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_dice = []
    for r, c in zip(row_ind, col_ind):
        d = float(dice_mat[r, c])
        if min_dice > 0.0 and d < min_dice:
            continue
        matched_dice.append(d)
    
    print(matched_dice)
    return float(np.mean(matched_dice)) if matched_dice else float("nan")


if __name__ == "__main__":

    for i in range(3, 4):

        mask2 = load_itk(f'C:/Users/Jan/Desktop/tooth_fairy_challenge/evaluation/evaluation_results/dl_separation_{i}_new.nii.gz')
        mask1 = load_itk(f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{i}.nii.gz')

        dice = mean_dice_teeth_hungarian(mask1, mask2)
        print(f"Number {i} - Mean Dice: {dice}")