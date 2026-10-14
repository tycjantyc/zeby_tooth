"""Run one segmentation model and preserve the source NIfTI geometry."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from inference.pipeline import _load_model
from inference.volume import read_volume, save_mask
from processing.utils import norm_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", default="evaluation/evaluation_results/segmentation.nii.gz")
    parser.add_argument("--normalization", default="01", choices=["01", "minmax", "none"])
    parser.add_argument("--num_classes", type=int, help="Optional check against checkpoint channels")
    parser.add_argument("--model_input", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--model", choices=["hyspark", "resunet3dse"], help="Optional architecture check")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compression", help="Legacy compatibility option; output classes come from weights")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, inputs, classes = _load_model(args.model_weights, device)
    if args.num_classes is not None and args.num_classes != classes:
        parser.error(f"Checkpoint has {classes} output channels")
    architecture = "hyspark" if model.__class__.__name__ == "Hybird" else "resunet3dse"
    if args.model and args.model != architecture:
        parser.error(f"Checkpoint architecture is {architecture}")
    if inputs not in (1, 4):
        parser.error("Supported checkpoints have 1 intensity channel or 4 channels with coordinates")
    if architecture == "hyspark" and args.model_input != 96:
        parser.error("Bundled HySpark checkpoints require --model_input 96")
    if args.file.lower().endswith(".npy"):
        reference = sitk.GetImageFromArray(np.load(args.file))
    else:
        reference, _ = read_volume(args.file)
    output = Path(args.output)
    if output.resolve() == Path(args.file).resolve():
        parser.error("Output path must differ from the input")
    if inputs == 4:
        from evaluation.evaluate_agh_segmentor import evaluate_agh
        evaluate = evaluate_agh
    else:
        from evaluation.evaluate import evaluate
    labels = evaluate(model=model, file=args.file,
                      model_input_shape=(args.model_input,) * 3,
                      stride=(args.stride,) * 3, num_classes=classes,
                      normalization_function=norm_factory(args.normalization), device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(save_mask(labels, reference, output))


if __name__ == "__main__":
    main()
