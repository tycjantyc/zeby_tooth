import numpy as np
import sys
import SimpleITK as sitk
import os
sys.path.append(os.getcwd().removesuffix('\\evaluation'))
print(sys.path)
import nibabel as nib
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from models.class_segmentor_model import build_class_segmentor_model
import torch as tc
import torch.nn as nn
from processing.tooth_fairy_classes import teeth_instance, only_canal_classes
from processing.utils import load_itk, crop_to_mask
from processing.utils import embed_in_center
from tqdm import tqdm
from monai.losses import DiceLoss
from processing.utils import infill_label


def sliding_window_slices(size: int, window: int, stride: int) -> list[tuple[int, int]]:
    if size <= window:
        return [(0, size)]

    slices = []
    start = 0
    while True:
        end = start + window
        if end >= size:
            slices.append((size - window, size))
            break
        slices.append((start, end))
        start += stride

    deduped = []
    for rng in slices:
        if not deduped or deduped[-1] != rng:
            deduped.append(rng)
    return deduped


def eval(model, file_cbct, file_instance_seg, file_gt, out, infill=False):
    
    THRESHOLD = 0.5
    LOSS = DiceLoss(include_background=True, sigmoid=False)
    
    image = load_itk(file_cbct)
    helper = load_itk(file_instance_seg)
    if file_gt is not None:
        ground_truth = load_itk(file_gt)

    print(image.shape)
    print(helper.shape)

    if infill:
        for teeth_num in tqdm(np.unique(helper)[1:]):
            helper = infill_label(helper, target=teeth_num, closing_iters=4, background=0, method='infill')

    canal_mask = np.zeros(image.shape)

    for teeth_num in tqdm(np.unique(helper)[1:]):

        temp_mask = np.zeros(helper.shape)

        temp_mask[helper != teeth_num] = 0
        temp_mask[helper == teeth_num] = 1

        image_snipped, mask_snipped, coords = crop_to_mask(image, temp_mask, padding=0, input=1, coordss=True)
        if file_gt is not None:
            gt_snipped, _= crop_to_mask(ground_truth, temp_mask, padding=0, input=1, coordss=False)
        

        TILE_SIZE = 96
        TILE_STRIDE = TILE_SIZE // 2

        agg_prediction = np.zeros_like(image_snipped, dtype=np.float32)
        agg_counts = np.zeros_like(image_snipped, dtype=np.float32)

        z_slices = sliding_window_slices(image_snipped.shape[0], TILE_SIZE, TILE_STRIDE)
        y_slices = sliding_window_slices(image_snipped.shape[1], TILE_SIZE, TILE_STRIDE)
        x_slices = sliding_window_slices(image_snipped.shape[2], TILE_SIZE, TILE_STRIDE)

        for z_start, z_end in z_slices:
            for y_start, y_end in y_slices:
                for x_start, x_end in x_slices:
                    image_tile = image_snipped[z_start:z_end, y_start:y_end, x_start:x_end]
                    mask_tile = mask_snipped[z_start:z_end, y_start:y_end, x_start:x_end]

                    image_tile_embed, embed_coords = embed_in_center(image_tile, TILE_SIZE, coords=True)
                    mask_tile_embed = embed_in_center(mask_tile, TILE_SIZE, coords=False)

                    masked = image_tile_embed[mask_tile_embed > 0]
                    if masked.size > 0:
                        image_tile_embed[mask_tile_embed > 0] = (masked - masked.min()) / (masked.max() - masked.min() + 1e-5)
                    image_tile_embed[mask_tile_embed < 1] = 0

                    tile_tensor = tc.tensor(image_tile_embed[None, None, ...], dtype=tc.float32).cuda()

                    with tc.no_grad():
                        pred_tile = nn.Sigmoid()(model(tile_tensor))

                    if file_gt is not None:
                        gt_tile = gt_snipped[z_start:z_end, y_start:y_end, x_start:x_end]
                        gt_tile_embed = embed_in_center(gt_tile, TILE_SIZE, coords=False)
                        gt_tile_tensor = tc.tensor(gt_tile_embed[None, None, ...], dtype=tc.float32).cuda()
                        dice_loss = LOSS(pred_tile, gt_tile_tensor)
                        print(f"Teeth number {teeth_num} chunk ({z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}) - Dice loss: {dice_loss.item()}")

                    start, end = embed_coords
                    pred_tile_np = pred_tile.detach().cpu().numpy()[0, 0]
                    pred_tile_np = pred_tile_np[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

                    agg_prediction[z_start:z_end, y_start:y_end, x_start:x_end] += pred_tile_np
                    agg_counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1

        aggregated_output = np.zeros_like(agg_prediction, dtype=np.float32)
        nonzero = agg_counts > 0
        aggregated_output[nonzero] = agg_prediction[nonzero] / agg_counts[nonzero]

        if file_gt is not None:
            dice_loss = LOSS(tc.tensor(aggregated_output[None, None, ...], dtype=tc.float32).cuda(),
                             tc.tensor(gt_snipped[None, None, ...], dtype=tc.float32).cuda())
            print(f"Teeth number {teeth_num} - Aggregated Dice loss: {dice_loss.item()}")

        output = np.zeros_like(aggregated_output, dtype=np.uint8)
        output[aggregated_output > THRESHOLD] = 1

        z_min, z_max, y_min, y_max, x_min, x_max = coords

        canal_mask_temp = np.zeros(canal_mask.shape)
        canal_mask_temp[z_min:z_max, y_min:y_max, x_min:x_max] = output
        canal_mask[canal_mask_temp == 1] = 1

        tc.cuda.empty_cache()

    canal_mask[helper == 0] = 0
    nifti_img = sitk.GetImageFromArray(canal_mask.astype(np.uint8))
    original = sitk.ReadImage(file_instance_seg)
    nifti_img.CopyInformation(original)

    sitk.WriteImage(nifti_img, out)
    
    if file_gt is not None:
        def dice(mask1, mask2):
            tp = np.count_nonzero((mask1+mask2) == 2)
            fp = np.count_nonzero((mask1+mask2) == 1)

            return 2*tp/(2*tp+fp)

        print(f"Dice score: {dice(canal_mask, ground_truth)}")

if __name__ == "__main__":

    PATH1 = '../../weights_tooth_fairy/tooth_fairy_3d_best_agh.pt'
    FILE = 20
    
    for i in range(1):
        
        model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False, weight_path=PATH1)
        model.eval()
        model = model.cuda()
        
        eval(model=model, 
            file_cbct=f'D:/db1_v1.0/db1_v1.0_nifti/{FILE}/image.nii.gz', 
            file_instance_seg = f'D:/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{FILE}.nii.gz',
            #file_instance_seg = f'C:/Users/Jan/Downloads/{FILE}.nii.gz',
            file_gt=None, #f'D:/db1_v1.0/db1_v1.0_nifti_root_canal_masks_single_cls_ref/10.nii.gz', 
            out = f'../evaluation/evaluation_results/output_canals_mask_{FILE}_test.nii.gz')
        
        FILE += 1
