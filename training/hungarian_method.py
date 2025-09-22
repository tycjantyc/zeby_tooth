import torch
from torch import nn
from typing import List, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except:
    HAS_SCIPY = False


def pairwise_dice_cost_3d(pred_masks: torch.Tensor, tgt_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_masks: (Q, D, H, W)  - LOGITS (sigmoid w środku)
    tgt_masks:  (G, D, H, W)  - binarne {0,1}
    Zwraca: (Q, G) macierz kosztów = 1 - soft Dice
    """
    Q, D, H, W = pred_masks.shape
    G = tgt_masks.shape[0]

    p = pred_masks.sigmoid().reshape(Q, -1)  # (Q, DHW)
    g = tgt_masks.float().reshape(G, -1)     # (G, DHW)

    # (Q,1,N) * (1,G,N) -> (Q,G,N) -> sum_N
    intersect = (p[:, None, :] * g[None, :, :]).sum(-1)    # (Q,G)
    p_sum     = p.sum(-1)[:, None]                         # (Q,1)
    g_sum     = g.sum(-1)[None, :]                         # (1,G)

    dice = (2.0 * intersect + eps) / (p_sum + g_sum + eps) # (Q,G)
    return 1.0 - dice


def hungarian_match_dice_3d(pred_masks: torch.Tensor, tgt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dopasowanie dla jednego wolumenu.
    pred_masks: (Q, D, H, W)
    tgt_masks:  (G, D, H, W)
    Zwraca indeksy (pred_ind, tgt_ind) w dopasowaniu Hungarian.
    """
    Q = pred_masks.shape[0]
    G = tgt_masks.shape[0]
    if Q == 0 or G == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    with torch.no_grad():
        cost = pairwise_dice_cost_3d(pred_masks, tgt_masks).detach().cpu()  # (Q,G)

        if HAS_SCIPY:
            r_idx, c_idx = linear_sum_assignment(cost.numpy())
        else:
            # Zachłanne awaryjne (nieoptymalne, ale bez SciPy)
            cost_np = cost.numpy()
            r_idx, c_idx = [], []
            used_rows, used_cols = set(), set()
            while len(used_rows) < min(Q, G):
                i, j = np.unravel_index(np.argmin(cost_np), cost_np.shape)
                if i not in used_rows and j not in used_cols:
                    r_idx.append(i); c_idx.append(j)
                    used_rows.add(i); used_cols.add(j)
                cost_np[i, :] = np.inf
                cost_np[:, j] = np.inf

        return torch.as_tensor(r_idx, dtype=torch.long), torch.as_tensor(c_idx, dtype=torch.long)

class DiceLoss3D(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, D, H, W)
        targets: (N, D, H, W) binarne {0,1}
        """
        probs = logits.sigmoid().reshape(logits.size(0), -1)   # (N, DHW)
        targets = targets.float().reshape(targets.size(0), -1) # (N, DHW)

        intersect = (probs * targets).sum(-1)
        denom = probs.sum(-1) + targets.sum(-1)
        dice = (2*intersect + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


class HungarianDiceCriterion3D(nn.Module):
    """
    Kryterium: tylko Dice. Stała liczba slotów Q na wolumen.
    Wejście: pred_masks (B,Q,D,H,W) - LOGITS
             gt_masks: lista długości B; element b to tensor (G_b,D,H,W) binarny
    """
    def __init__(self, eps: float = 1e-6, lambda_unmatched: float = 1.0):
        super().__init__()
        self.dice = DiceLoss3D(eps)
        self.lambda_unmatched = lambda_unmatched

    def forward(self, pred_masks: torch.Tensor, gt_masks: List[torch.Tensor]) -> float:
        device = pred_masks.device
        B, Q, D, H, W = pred_masks.shape
        dice_losses = []
        unmatched_losses = []

        for b in range(B):
            pm = pred_masks[b]                # (Q,D,H,W)
            gtb = gt_masks[b].to(device)      # (G,D,H,W) lub (0,D,H,W)

            pred_idx, tgt_idx = hungarian_match_dice_3d(pm, gtb)

            # 1) matched pairs
            if len(pred_idx) > 0:
                matched_pred = pm[pred_idx]    # (M,D,H,W)
                matched_tgt  = gtb[tgt_idx]    # (M,D,H,W)
                dice_losses.append(self.dice(matched_pred, matched_tgt))

            # 2) unmatched predictions -> Dice względem pustej maski
            all_pred = torch.arange(Q, device=device)
            if len(pred_idx) > 0:
                mask = torch.ones(Q, dtype=torch.bool, device=device)
                mask[pred_idx.to(device)] = False
                unmatched_pred_idx = all_pred[mask]
            else:
                unmatched_pred_idx = all_pred

            if len(unmatched_pred_idx) > 0:
                up = pm[unmatched_pred_idx]                 # (U,D,H,W)
                zeros = torch.zeros_like(up)                # pusta maska
                unmatched_losses.append(self.dice(up, zeros))

        loss_mask = 0.0
        if len(dice_losses) > 0:
            loss_mask = loss_mask + torch.stack(dice_losses).mean()
        if len(unmatched_losses) > 0:
            loss_mask = loss_mask + self.lambda_unmatched * torch.stack(unmatched_losses).mean()

        return loss_mask
