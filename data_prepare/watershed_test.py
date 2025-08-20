#!/usr/bin/env python3
"""
teeth_watershed.py

Marker-based 3D watershed from class centers to (re)separate teeth.
- Seeds: one per input label (>0), either geometric centroids or distance-transform peaks.
- Floods on the NEGATIVE distance map so basins start at object centers.
- Preserves input image metadata (spacing, origin, direction) in the output NIfTI.

Usage:
  python teeth_watershed.py \
      --in-mask path/to/labels.nii.gz \
      --out path/to/labels_watershed.nii.gz \
      [--seed-mode {peak,centroid}] \
      [--bg 0] [--fully-connected] [--save-centers centers.csv] [--reference path/to/reference.nii.gz]

Requires: SimpleITK, numpy
"""

import argparse
import csv
import sys
from typing import Dict, Tuple, List

import numpy as np
import SimpleITK as sitk


def get_unique_labels(img: sitk.Image, bg: int) -> List[int]:
    arr = sitk.GetArrayViewFromImage(img)
    labels = np.unique(arr)
    labels = labels[labels != bg]
    return labels.tolist()


def label_mask(img: sitk.Image, label_value: int) -> sitk.Image:
    return sitk.Equal(img, int(label_value))


def compute_distance(binary_mask: sitk.Image) -> sitk.Image:
    """
    Signed Maurer distance: inside positive (distance to background), outside negative.
    We will negate it later for watershed basins at object centers.
    """
    dist = sitk.SignedMaurerDistanceMap(
        binary_mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return dist


def seed_from_dt_peak(lbl_img: sitk.Image, dist: sitk.Image, label_value: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    One seed per label: argmax of distance transform inside that label (robustly interior).
    Returns (index_xyz, physical_xyz)
    """
    m = label_mask(lbl_img, label_value)
    # Mask the distance so only current label contributes (set outside to very low)
    masked = sitk.Mask(dist, m, outsideValue=-1e9)
    a = sitk.GetArrayViewFromImage(masked)  # (z,y,x)
    k = int(np.argmax(a))
    z, y, x = np.unravel_index(k, a.shape)
    idx = (int(x), int(y), int(z))  # SimpleITK index order
    phys = lbl_img.TransformIndexToPhysicalPoint(idx)
    return idx, phys


def seed_from_centroid(lbl_img: sitk.Image, label_value: int) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    """
    Geometric centroid in physical coords mapped to nearest index.
    If rounding lands outside the label (rare), fall back to DT peak.
    """
    lss = sitk.LabelShapeStatisticsImageFilter()
    lss.Execute(sitk.RelabelComponent(sitk.ConnectedComponent(label_mask(lbl_img, label_value))))
    # Merge all components of this label mask before centroid: simpler to compute centroid directly
    # Use sitk’s built-in centroid on a single-label image:
    # Create a temporary binary image for this label with pixel value 1
    bin_img = label_mask(lbl_img, label_value)
    # LabelShapeStatistics expects labeled image; label the single region
    comp = sitk.ConnectedComponent(bin_img)
    lss2 = sitk.LabelShapeStatisticsImageFilter()
    lss2.Execute(comp)
    # Use the largest component’s centroid if there are multiple
    largest = max(lss2.GetLabels(), key=lambda lab: lss2.GetPhysicalSize(lab))
    centroid_phys = lss2.GetCentroid(largest)  # (x,y,z) in physical
    idx = lbl_img.TransformPhysicalPointToIndex(centroid_phys)

    # If that voxel doesn't belong to the label (rounding issues), fallback to DT peak
    if int(sitk.GetArrayViewFromImage(lbl_img)[idx[2], idx[1], idx[0]]) != int(label_value):
        # Compute DT once (cheap for single label)
        dist = compute_distance(bin_img)
        return seed_from_dt_peak(lbl_img, dist, label_value)
    return idx, centroid_phys


def build_marker_image(
    ref_img: sitk.Image,
    seeds: Dict[int, Tuple[int, int, int]],
    dtype=sitk.sitkUInt16,
) -> sitk.Image:
    markers = sitk.Image(ref_img.GetSize(), dtype)
    markers.CopyInformation(ref_img)
    # Put label ids directly as marker values (keeps original IDs if they fit dtype)
    for lab, idx in seeds.items():
        # Safety: clamp to image bounds
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


def main():
    ap = argparse.ArgumentParser(description="3D watershed from per-label centers (teeth).")
    ap.add_argument("--in-mask", required=True, help="Path to input multi-class label NIfTI (.nii/.nii.gz).")
    ap.add_argument("--out", required=True, help="Path to output NIfTI.")
    ap.add_argument("--bg", type=int, default=0, help="Background label value. Default: 0")
    ap.add_argument("--seed-mode", choices=["peak", "centroid"], default="peak",
                    help="Seed type per label: 'peak' (DT maximum within label) or 'centroid'. Default: peak")
    ap.add_argument("--fully-connected", action="store_true",
                    help="Use 26-connectivity in 3D (SimpleITK fullyConnected=True).")
    ap.add_argument("--save-centers", default=None, help="Optional CSV to save centers (label,x,y,z in physical units).")
    ap.add_argument("--reference", default=None,
                    help="Optional NIfTI to borrow metadata from (spacing, origin, direction). Default: use in-mask.")
    args = ap.parse_args()

    lbl = sitk.ReadImage(args.in_mask)
    if args.reference:
        ref = sitk.ReadImage(args.reference)
    else:
        ref = lbl

    # Binary mask of all teeth (anything != bg)
    binary = sitk.NotEqual(lbl, int(args.bg))

    # Distance transform once for peak seeds (faster than per-label)
    if args.seed_mode == "peak":
        dist_full = compute_distance(binary)
    else:
        dist_full = None  # computed per-label if needed

    labels = get_unique_labels(lbl, args.bg)
    if len(labels) == 0:
        print("No foreground labels found. Exiting.", file=sys.stderr)
        sys.exit(1)

    seeds_idx: Dict[int, Tuple[int, int, int]] = {}
    centers_phys: Dict[int, Tuple[float, float, float]] = {}

    for lab in labels:
        if args.seed_mode == "peak":
            # Use the global distance map but mask per label to find the peak
            idx, phys = seed_from_dt_peak(lbl, dist_full, lab)
        else:
            idx, phys = seed_from_centroid(lbl, lab)
        seeds_idx[lab] = idx
        centers_phys[lab] = phys

    # Build marker image with original label IDs as marker values
    markers = build_marker_image(ref, seeds_idx, dtype=sitk.sitkUInt16)

    # Watershed on negative distance
    if dist_full is None:
        dist_full = compute_distance(binary)
    ws = watershed_from_markers(dist_full, markers, fully_connected=args.fully_connected)

    # Restrict to tooth mask (set background to 0 outside)
    ws_masked = sitk.Mask(ws, binary, outsideValue=int(args.bg))

    # Ensure output inherits full spatial metadata from reference
    ws_masked.CopyInformation(ref)

    # Write output
    sitk.WriteImage(ws_masked, args.out)
    print(f"Saved watershed labels to: {args.out}")

    # Optionally save centers CSV
    if args.save_centers:
        with open(args.save_centers, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "x_mm", "y_mm", "z_mm"])
            for lab in labels:
                x, y, z = centers_phys[lab]
                w.writerow([int(lab), x, y, z])
        print(f"Saved centers to: {args.save_centers}")

if __name__ == "__main__":
    main()
