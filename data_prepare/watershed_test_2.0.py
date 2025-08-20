#!/usr/bin/env python3
"""
teeth_watershed_dual_input.py

Compute centers from one image (--seed-mask) and apply marker-based 3D watershed
on another image (--ws-mask). Output inherits spacing/origin/direction from the
watershed image.

Usage:
  python teeth_watershed_dual_input.py \
      --seed-mask path/to/instances_or_multilabel.nii.gz \
      --ws-mask   path/to/semantic_teeth_mask.nii.gz \
      --out       path/to/labels_watershed.nii.gz \
      [--seed-mode {peak,centroid}] \
      [--seed-bg 0] [--ws-bg 0] \
      [--snap-radius 6] [--no-snap] \
      [--fully-connected] \
      [--save-centers centers.csv]

Requires: SimpleITK, numpy
"""

import argparse
import csv
import sys
from typing import Dict, Tuple, List

import numpy as np
import SimpleITK as sitk


# ------------------------- utilities -------------------------

def get_unique_labels(img: sitk.Image, bg: int) -> List[int]:
    arr = sitk.GetArrayViewFromImage(img)
    labels = np.unique(arr)
    labels = labels[labels != bg]
    return labels.tolist()


def label_mask(img: sitk.Image, label_value: int) -> sitk.Image:
    return sitk.Equal(img, int(label_value))


def compute_distance(binary_mask: sitk.Image, use_spacing: bool = True) -> sitk.Image:
    """
    Signed Maurer distance: inside positive (distance to background), outside negative.
    """
    dist = sitk.SignedMaurerDistanceMap(
        binary_mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=use_spacing,
    )
    return dist


def seed_from_dt_peak(lbl_img: sitk.Image, label_value: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    One seed per label: argmax of distance transform INSIDE that label (robustly interior).
    Returns (index_xyz_in_lbl_img, physical_xyz)
    """
    bin_img = label_mask(lbl_img, label_value)
    dist = compute_distance(bin_img, use_spacing=True)
    # Mask the distance so only current label contributes (set outside to very low)
    masked = sitk.Mask(dist, bin_img, outsideValue=-1e9)
    a = sitk.GetArrayViewFromImage(masked)  # (z,y,x)
    if a.size == 0 or np.all(a < -1e8):
        raise RuntimeError(f"No voxels found for label {label_value}")
    k = int(np.argmax(a))
    z, y, x = np.unravel_index(k, a.shape)
    idx = (int(x), int(y), int(z))  # SimpleITK index order
    phys = lbl_img.TransformIndexToPhysicalPoint(idx)
    return idx, phys


def seed_from_centroid(lbl_img: sitk.Image, label_value: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Geometric centroid (physical) mapped to nearest index. If rounding lands outside,
    fall back to DT peak.
    """
    bin_img = label_mask(lbl_img, label_value)
    comp = sitk.ConnectedComponent(bin_img)
    lss = sitk.LabelShapeStatisticsImageFilter()
    lss.Execute(comp)
    if len(lss.GetLabels()) == 0:
        raise RuntimeError(f"No components found for label {label_value}")
    # Use largest component
    largest = max(lss.GetLabels(), key=lambda lab: lss.GetPhysicalSize(lab))
    centroid_phys = lss.GetCentroid(largest)  # (x,y,z) physical
    idx = lbl_img.TransformPhysicalPointToIndex(centroid_phys)

    # If not inside the label, fallback to DT peak
    arr = sitk.GetArrayViewFromImage(lbl_img)
    if not (0 <= idx[0] < arr.shape[2] and 0 <= idx[1] < arr.shape[1] and 0 <= idx[2] < arr.shape[0]) \
       or int(arr[idx[2], idx[1], idx[0]]) != int(label_value):
        return seed_from_dt_peak(lbl_img, label_value)
    return idx, centroid_phys


def build_marker_image(
    ref_img: sitk.Image,
    seeds: Dict[int, Tuple[int, int, int]],
    dtype=sitk.sitkUInt32,
) -> sitk.Image:
    markers = sitk.Image(ref_img.GetSize(), dtype)
    markers.CopyInformation(ref_img)
    for lab, idx in seeds.items():
        if all(0 <= idx[d] < ref_img.GetSize()[d] for d in range(3)):
            markers[idx] = int(lab)
    return markers


def watershed_from_markers(dist: sitk.Image, markers: sitk.Image, fully_connected: bool) -> sitk.Image:
    neg_dist = sitk.ShiftScale(dist, shift=0.0, scale=-1.0)
    ws = sitk.MorphologicalWatershedFromMarkers(
        image=neg_dist,
        markerImage=markers,
        markWatershedLine=False,
        fullyConnected=fully_connected,
    )
    return ws


def find_nearest_foreground_index(ws_arr: np.ndarray, idx_xyz: Tuple[int, int, int],
                                  spacing_zyx: Tuple[float, float, float],
                                  max_radius: int = 6):
    """
    Brute-force small-radius search around idx to find the closest foreground voxel (True in ws_arr!=0).
    idx_xyz is in SimpleITK index order (x,y,z). Array is (z,y,x).
    Returns array-order index (z,y,x) converted back to SITK index (x,y,z), or None if not found.
    """
    zyx = (idx_xyz[2], idx_xyz[1], idx_xyz[0])
    Z, Y, X = ws_arr.shape
    if not (0 <= zyx[0] < Z and 0 <= zyx[1] < Y and 0 <= zyx[2] < X):
        return None

    if ws_arr[zyx]:
        return idx_xyz  # already on foreground

    best = None
    best_d2 = float("inf")
    sz, sy, sx = spacing_zyx
    for r in range(1, max_radius + 1):
        z0, z1 = max(0, zyx[0]-r), min(Z, zyx[0]+r+1)
        y0, y1 = max(0, zyx[1]-r), min(Y, zyx[1]+r+1)
        x0, x1 = max(0, zyx[2]-r), min(X, zyx[2]+r+1)
        sub = ws_arr[z0:z1, y0:y1, x0:x1]
        if not sub.any():
            continue
        # coordinates of foreground within the subvolume
        zz, yy, xx = np.nonzero(sub)
        # convert to absolute coords
        zz += z0; yy += y0; xx += x0
        # squared physical distance
        dz = (zz - zyx[0]) * sz
        dy = (yy - zyx[1]) * sy
        dx = (xx - zyx[2]) * sx
        d2 = dz*dz + dy*dy + dx*dx
        k = int(np.argmin(d2))
        if d2[k] < best_d2:
            best_d2 = float(d2[k])
            best = (int(xx[k]), int(yy[k]), int(zz[k]))  # (x,y,z) in array order swapped later
            # early stop if extremely close
            if best_d2 == 0:
                break
    if best is None:
        return None
    # convert back to SITK index order (x,y,z) from array order (z,y,x)
    return (best[0], best[1], best[2])


# ------------------------- main -------------------------

def main(seed_mask, ws_mask, out, ws_big, seed_mode, fully_connected, no_snap, snap_radius):
    

    seed_img = sitk.ReadImage(seed_mask)
    ws_img   = sitk.ReadImage(ws_mask)

    # Foreground (tooth) mask for watershed image
    ws_arr = sitk.GetArrayFromImage(ws_img)  # (z,y,x)
    ws_fg = (ws_arr != ws_big)

    # Collect labels from seed image (ignore background)
    labels = get_unique_labels(seed_img, 0)
    if len(labels) == 0:
        print("No foreground labels found in --seed-mask. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Compute seeds in seed image space (physical coordinates)
    seeds_phys: Dict[int, Tuple[float, float, float]] = {}
    seeds_idx_ws: Dict[int, Tuple[int, int, int]] = {}

    for lab in labels:
        if seed_mode == "peak":
            _, phys = seed_from_dt_peak(seed_img, lab)
        else:
            _, phys = seed_from_centroid(seed_img, lab)
        seeds_phys[lab] = phys

        # Map to watershed image index space via physical coordinates
        idx_ws = ws_img.TransformPhysicalPointToIndex(phys)

        # If out of bounds, drop (or try to clamp?)
        in_bounds = all(0 <= idx_ws[d] < ws_img.GetSize()[d] for d in range(3))
        if not in_bounds:
            # skip this seed
            continue

        # Optional snap to nearest foreground if current voxel is background
        zyx = (idx_ws[2], idx_ws[1], idx_ws[0])
        if not no_snap and not ws_fg[zyx]:
            spacing = ws_img.GetSpacing()  # (sx,sy,sz) -> array order (sz,sy,sx)
            snapped = find_nearest_foreground_index(ws_fg, idx_ws, (spacing[2], spacing[1], spacing[0]),
                                                    max_radius=snap_radius)
            if snapped is not None:
                idx_ws = snapped
            else:
                # couldn't find; skip this seed
                continue

        seeds_idx_ws[lab] = idx_ws

    if len(seeds_idx_ws) == 0:
        print("No valid seeds landed inside the watershed mask. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build marker image in the watershed image's space
    markers = build_marker_image(ws_img, seeds_idx_ws, dtype=sitk.sitkUInt32)

    # Distance transform of the watershed foreground (teeth)
    ws_binary = sitk.NotEqual(ws_img, int(ws_big))
    dist_ws = compute_distance(ws_binary, use_spacing=True)

    # Watershed on negative distance
    ws_labels = watershed_from_markers(dist_ws, markers, fully_connected = fully_connected)

    # Keep only inside the watershed foreground
    ws_labels = sitk.Mask(ws_labels, ws_binary, outsideValue=int(ws_big))
    ws_labels.CopyInformation(ws_img)

    # Save output
    sitk.WriteImage(ws_labels, out)
    print(f"Saved watershed labels to: {out}")

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="3D watershed with centers from one image and flooding on another.")
    ap.add_argument("--seed-mask", required=True, help="Path to input multi-class label NIfTI for center extraction.")
    ap.add_argument("--ws-mask",   required=True, help="Path to semantic (teeth) mask NIfTI for watershed.")
    ap.add_argument("--out",       required=True, help="Path to output NIfTI.")
    ap.add_argument("--ws-bg",   type=int, default=0, help="Background value in ws-mask (non-foreground). Default: 0")
    ap.add_argument("--seed-mode", choices=["peak", "centroid"], default="peak",
                    help="Seed type per label from seed-mask: 'peak' (DT maximum within label) or 'centroid'. Default: peak")
    ap.add_argument("--fully-connected", action="store_true",
                    help="Use 26-connectivity in 3D for watershed.")
    ap.add_argument("--no-snap", action="store_true",
                    help="Disable snapping seeds that fall on ws-bg to nearest foreground.")
    ap.add_argument("--snap-radius", type=int, default=6,
                    help="Max search radius (voxels) when snapping seeds to ws foreground (if needed). Default: 6")
    args = ap.parse_args()


    main(args.seed_mask, args.ws_mask, args.out, args.ws_big, args.seed_mode, args.fully_connected, args.no_snap, args.snap_radius)