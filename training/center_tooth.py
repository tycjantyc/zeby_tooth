import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.measure import label
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

import numpy as np

def centroids_3d(labels: np.ndarray, background: int = 0):
    """
    labels: (D,H,W) int
    Zwraca:
      ids: (K,) label id (bez tła)
      ctr: (K,3) centroidy w kolejności (z, y, x) jako float
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels musi być 3D (D,H,W)")

    ids = np.unique(labels)
    ids = ids[ids != background]
    if ids.size == 0:
        return ids, np.zeros((0, 3), dtype=np.float32)

    # spłaszczamy i bierzemy współrzędne nie-tła
    flat = labels.ravel()
    idx = np.nonzero(flat != background)[0]
    lab = flat[idx]

    zyx = np.column_stack(np.unravel_index(idx, labels.shape)).astype(np.float64)  # (N,3)

    # map label->0..K-1
    order = np.argsort(ids)
    ids_sorted = ids[order]
    inv = np.searchsorted(ids_sorted, lab)

    # sumy współrzędnych i liczności
    K = ids_sorted.size
    counts = np.bincount(inv, minlength=K).astype(np.float64)
    sums_z = np.bincount(inv, weights=zyx[:, 0], minlength=K)
    sums_y = np.bincount(inv, weights=zyx[:, 1], minlength=K)
    sums_x = np.bincount(inv, weights=zyx[:, 2], minlength=K)

    ctr = np.stack([sums_z / counts, sums_y / counts, sums_x / counts], axis=1).astype(np.float32)
    return ids_sorted, ctr

def match_by_nearest_centroid(pred, gt, max_dist=10.0, background=0):
    output = torch.sigmoid(pred)
    #output = output*255
    output[output <= 0.5] = 0
    output[output > 0.5] = 1

    output = output[0, 0].detach().cpu().numpy().astype(np.uint16)
    output = label(output)
    pred = output

    output_1 = gt
    output_1[output_1 <= 0.5] = 0
    output_1[output_1 > 0.5] = 1

    output_1 = output_1[0, 0].detach().cpu().numpy().astype(np.uint16)
    output_1 = label(output_1)
    gt = output_1

    pred_ids, pred_ctr = centroids_3d(pred, background)
    gt_ids, gt_ctr = centroids_3d(gt, background)

    if pred_ids.size == 0:
        return {
            "unmatched_pred": 0,
            "unmatched_gt": int(gt_ids.size),
            "pairs": []
        }
    if gt_ids.size == 0:
        return {
            "unmatched_pred": int(pred_ids.size),
            "unmatched_gt": 0,
            "pairs": []
        }

    # odległości (Kp, Kg)
    diff = pred_ctr[:, None, :] - gt_ctr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    nn_j = dist.argmin(axis=1)
    nn_d = dist[np.arange(dist.shape[0]), nn_j]

    pairs = []
    matched_pred = np.zeros(pred_ids.size, dtype=bool)
    matched_gt = np.zeros(gt_ids.size, dtype=bool)

    for i, (j, d) in enumerate(zip(nn_j, nn_d)):
        if d <= max_dist:
            pairs.append((int(pred_ids[i]), int(gt_ids[j]), float(d)))
            matched_pred[i] = True
            matched_gt[j] = True  # uwaga: wiele pred może ustawić ten sam gt na True

    return {
        "unmatched_pred": int((~matched_pred).sum()),
        "unmatched_gt": int((~matched_gt).sum()),
        "pairs": pairs
    }

class CenterNetHeatmapLoss(nn.Module):
    """
    CenterNet/CornerNet modified focal loss for center heatmaps.
    Works for 2D or 3D heatmaps (any spatial dims).

    Inputs:
      pred_logits: (B, 1, *spatial) raw logits
      gt:         (B, 1, *spatial) in [0,1], made from Gaussians,
                  with exact center voxels set to 1.0

    Args:
      alpha: exponent for modulating factor on positives/negatives (default 2)
      beta:  exponent for negative weights (1 - gt)^beta (default 4)
      eps:   numerical stability
      normalize_by_num_pos: divide by number of positive centers (default True)
    """
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-6,
        normalize_by_num_pos: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.normalize_by_num_pos = normalize_by_num_pos
        self.dice = DiceFocalLoss(sigmoid=True)

    def forward(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if pred_logits.shape != gt.shape:
            raise ValueError(f"Shape mismatch: pred {pred_logits.shape} vs gt {gt.shape}")

        # Probability map
        pred = torch.sigmoid(pred_logits).clamp(self.eps, 1.0 - self.eps)

        # Positives are exact centers (gt == 1). Everything else is treated as negative.
        pos_inds = (gt == 1).to(pred.dtype)
        neg_inds = (gt < 1).to(pred.dtype)

        # Down-weight negatives near positives (Gaussian shoulders)
        neg_weights = (1.0 - gt).clamp_min(0).pow(self.beta)

        pos_loss = -(pred.log()) * (1.0 - pred).pow(self.alpha) * pos_inds
        neg_loss = -((1.0 - pred).log()) * (pred).pow(self.alpha) * neg_weights * neg_inds

        loss = 5*pos_loss.sum() + neg_loss.sum()

        if self.normalize_by_num_pos:
            num_pos = pos_inds.sum().clamp_min(1.0)
            loss = loss / num_pos

        return loss #+ self.dice(pred_logits, gt) * 10 + mse_loss(pred_logits, gt) * 10


def instance_mask_to_heatmap_torch(instance_mask: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """
    Converts a PyTorch instance segmentation mask tensor into a heatmap tensor.

    Each instance on the mask is represented as a Gaussian blob on the output
    heatmap, centered at its centroid. This function is designed to be
    differentiable with respect to the input mask's values if needed, although
    the centroid calculation involves indexing which is non-differentiable.

    Args:
        instance_mask (torch.Tensor): Tensor of shape [B, C, D, H, W]
            containing the segmentation mask. Values are integers, where 0 is
            background and values > 0 are instance IDs. Assumes C=1.
        sigma (float): Standard deviation for the Gaussian kernel, controlling
            the size of the blobs on the heatmap. Defaults to 3.0.

    Returns:
        torch.Tensor: Heatmap tensor of shape [B, 1, D, H, W] and dtype float32,
            with values in the range [0, 1].
    """
    if instance_mask.dim() != 5:
        raise ValueError(f"Expected 5 dimensions [B, C, D, H, W], got {instance_mask.dim()}")
    
    if instance_mask.shape[1] != 1:
        print(f"Warning: Expected 1 channel (C=1), but shape is {instance_mask.shape}. "
              "Using the first channel only.")

    device = instance_mask.device
    batch_size = instance_mask.shape[0]
    spatial_dims = instance_mask.shape[2:]
    batch_heatmaps = []

    for i in range(batch_size):
        single_mask = instance_mask[i, 0, ...]
        
        instance_ids = torch.unique(single_mask)
        instance_ids = instance_ids[instance_ids != 0]

        if len(instance_ids) == 0:
            batch_heatmaps.append(torch.zeros(spatial_dims, device=device, dtype=torch.float32))
            continue
            
        point_map = torch.zeros(spatial_dims, device=device, dtype=torch.float32)

        for inst_id in instance_ids:
            # Find coordinates of all voxels for the current instance
            coords = torch.nonzero(single_mask == inst_id, as_tuple=False)
            if coords.numel() > 0:
                # Calculate the centroid (mean of coordinates)
                centroid = torch.mean(coords.float(), dim=0)
                cz, cy, cx = centroid.long()

                # Ensure coordinates are within bounds before placing the point
                if 0 <= cz < spatial_dims[0] and 0 <= cy < spatial_dims[1] and 0 <= cx < spatial_dims[2]:
                    point_map[cz, cy, cx] = 1.0
        
        # Apply 3D Gaussian blur using convolution
        # 1. Create the Gaussian kernel
        kernel_size = int(2 * round(3 * sigma) + 1)
        if kernel_size % 2 == 0: kernel_size += 1 # Ensure odd kernel size
        
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum() # Normalize
        
        # Reshape for 3D convolution: (out_channels, in_channels, D, H, W)
        kernel_3d = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        
        # 2. Apply convolution
        # Add batch and channel dimensions for conv3d
        point_map_unsqueezed = point_map.view(1, 1, *spatial_dims)
        padding = kernel_size // 2
        
        heatmap = F.conv3d(point_map_unsqueezed, kernel_3d, padding=padding).squeeze()

        # Normalize heatmap to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
            
        batch_heatmaps.append(heatmap)


    final_heatmaps = torch.stack(batch_heatmaps, dim=0)
    final_heatmaps = final_heatmaps.unsqueeze(1) # Add channel dimension
    
    

    return final_heatmaps

def instance_mask_to_heatmap(instance_mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    
    if instance_mask.ndim != 5:
        raise ValueError(f"Oczekiwano 5 wymiarów [B, C, D, H, W], otrzymano {instance_mask.ndim}")
    
    if instance_mask.shape[1] != 1:
        print(f"Ostrzeżenie: Oczekiwano 1 kanału (C=1), ale kształt to {instance_mask.shape}. "
              "Użyty zostanie tylko pierwszy kanał.")

    batch_size = instance_mask.shape[0]
    spatial_dims = instance_mask.shape[2:]
    batch_heatmaps = []

    # Przetwarzaj każdy element w batchu osobno
    for i in range(batch_size):
        # Wyodrębnij pojedynczy wolumen 3D, usuwając wymiar kanału
        single_mask = instance_mask[i, 0, ...]
        
        # Znajdź unikalne ID instancji (pomijając tło, czyli 0)
        instance_ids = np.unique(single_mask)
        instance_ids = instance_ids[instance_ids != 0]
        
        # Jeśli nie ma żadnych instancji, zwróć pustą heatmapę
        if len(instance_ids) == 0:
            batch_heatmaps.append(np.zeros(spatial_dims, dtype=np.float32))
            continue
            
        # Znajdź centroidy dla wszystkich instancji jednocześnie
        # Wynik będzie listą krotek (z, y, x)
        centroids = center_of_mass(
            input=single_mask, 
            labels=single_mask, 
            index=instance_ids
        )
        
        # Stwórz mapę punktową, gdzie każdy centroid to wartość 1
        point_map = np.zeros(spatial_dims, dtype=np.float32)
        
        for centroid in centroids:
            # centroid to krotka (z, y, x) - musimy przekonwertować na int
            cz, cy, cx = map(int, centroid)
            
            # Upewnij się, że współrzędne są w granicach macierzy
            if 0 <= cz < spatial_dims[0] and 0 <= cy < spatial_dims[1] and 0 <= cx < spatial_dims[2]:
                point_map[cz, cy, cx] = 1.0

        # Zastosuj filtr Gaussa, aby zamienić punkty w gładkie plamki
        heatmap = gaussian_filter(point_map, sigma=sigma)
        
        # Znormalizuj heatmapę do zakresu [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val
            
        batch_heatmaps.append(heatmap)

    # Połącz wszystkie heatmapy w jeden batch i dodaj wymiar kanału
    final_heatmaps = np.stack(batch_heatmaps, axis=0)
    final_heatmaps = np.expand_dims(final_heatmaps, axis=1)
    
    return final_heatmaps.astype(np.float32)

from accelerate import Accelerator
from tqdm import tqdm
import torch as tc
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from monai.losses import DiceFocalLoss, DiceLoss, FocalLoss
from training.hungarian_method import HungarianDiceCriterion3D

def train_segmentor(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs, clearml_task=None):

    CRITERION = CenterNetHeatmapLoss(alpha=1, beta=2.0) #DiceFocalLoss(include_background=False, sigmoid=True, softmax=False) #
    DEVICE = 'cuda:0'
    training_losses = []
    validation_losses = []
    gt_missed = []
    pred_missed = []

    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    model = model.to(DEVICE)
    global_step = 0
    global_loss = 100000
    best_weights_path = Path("models/weights/agh_centroid_best.pt")
    clearml_logger = clearml_task.get_logger() if clearml_task is not None else None
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), ncols=100, dynamic_ncols=True)
        progress_bar.set_description(f"Epoch {epoch}")
        loss_full = 0
        missed = 0

        #Training phase
        for step, (label_small, label_big) in enumerate(train_dataloader):


            if step == 0 and epoch == 0:
                print(label_small.min(), label_small.max())
                print(label_big.min(), label_big.max())


            clean_output = instance_mask_to_heatmap_torch(label_big[:, None, :, :, :], sigma=2)

            # print(clean_output.max())
            # print(tc.sum((clean_output == 1)*1)) #dane są okej!

            clean_input = label_small.float().to(DEVICE) #[:, None, :, :, :]
            clean_output = clean_output.to(DEVICE)

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                label_pred = model(clean_input)

                print(label_pred.max()) #Ważne, czy model faktycznie znajduje piki!

                loss = CRITERION(label_pred, clean_output)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                

            #progress_bar.update(1)
            #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            #progress_bar.set_postfix(**logs)
            #accelerator.log(logs, step=global_step)

            #za dużo logów, ewentualnie można odznaczyć

            global_step += 1

            loss_full += loss.item()
            if clearml_logger is not None and step % 10 == 0:
                clearml_logger.report_scalar("train", "loss_step", loss.item(), global_step)

        loss_full = loss_full/(len(train_dataloader)-missed)

        training_losses.append(loss_full)
        if clearml_logger is not None:
            clearml_logger.report_scalar("train", "loss_epoch", loss_full, epoch + 1)
            clearml_logger.report_scalar("train", "lr", lr_scheduler.get_last_lr()[0], epoch + 1)

        print(f'Epoch: {epoch + 1}, Loss: {loss_full:.7f}')
        loss_val_full = 0
        print(f"Pominięte: {missed}")

        missed = 0
        gt_miss = 0
        pred_miss = 0
        
        #Validation phase
        with tc.no_grad():
            for step, (label_small, label_big) in enumerate(val_dataloader):


                clean_output = instance_mask_to_heatmap_torch(label_big[:, None, :, :, :], sigma=2)

                clean_input = label_small.float().to(DEVICE)
                clean_output = clean_output.to(DEVICE)

                with accelerator.accumulate(model):
            
                    label_pred = model(clean_input)
                    loss = CRITERION(label_pred, clean_output)
                    
                loss_val_full += loss.item()

                res = match_by_nearest_centroid(label_pred, clean_output, max_dist=5.0)  

                print("Unmatched pred labels:", res["unmatched_pred"])
                print("Unmatched gt labels:", res["unmatched_gt"])
                print("Matched pairs (pred, gt, dist):", res["pairs"][:10])

                gt_miss += res["unmatched_gt"]
                pred_miss += res["unmatched_pred"]

        gt_missed.append(gt_miss)
        pred_missed.append(pred_miss)
        if clearml_logger is not None:
            clearml_logger.report_scalar("val", "unmatched_gt", gt_miss, epoch + 1)
            clearml_logger.report_scalar("val", "unmatched_pred", pred_miss, epoch + 1)

        if len(val_dataloader) > 0:
            loss_val_full = loss_val_full/(len(val_dataloader) - missed)

            validation_losses.append(loss_val_full)
            if clearml_logger is not None:
                clearml_logger.report_scalar("val", "loss_epoch", loss_val_full, epoch + 1)
            print(f'Epoch: {epoch + 1}, Validation Loss: {loss_val_full:.7f}')

        if loss_val_full < global_loss:
            tc.save(model.state_dict(), str(best_weights_path))
            print(f"Weights saved! - Epoch {epoch}")
            global_loss = loss_val_full
            if clearml_task is not None:
                try:
                    clearml_task.upload_artifact(
                        name="agh_centroid_best",
                        artifact_object=str(best_weights_path),
                        metadata={"epoch": int(epoch + 1), "val_loss": float(global_loss)},
                    )
                except Exception as ex:
                    print(f"ClearML artifact upload failed: {ex}")

        tc.cuda.empty_cache()
        print("Cache emptied!")
    
    plt.plot(range(1, num_epochs+1), training_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylim(0, np.median(training_losses) + 3*np.std(training_losses))
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('losses.png')
    if clearml_logger is not None:
        clearml_logger.report_matplotlib_figure(
            title="training_curves",
            series="losses",
            iteration=num_epochs,
            figure=plt.gcf(),
            report_image=True,
        )

    print(gt_missed)
    print(pred_missed)
