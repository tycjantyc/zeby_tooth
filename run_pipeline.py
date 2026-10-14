"""CLI for the reusable segmentation pipeline."""

import argparse
import json

from inference.pipeline import DEFAULT_WEIGHTS, SegmentationPipeline


def main():
    parser = argparse.ArgumentParser(description="CBCT teeth and canal segmentation to NIfTI")
    parser.add_argument("--cbct", required=True, help="Input .nii or .nii.gz")
    parser.add_argument("--output-dir", required=True, help="Empty result directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teeth-mask", help="Binary/integer mask of all teeth including pulp")
    source.add_argument("--instances", help="Instance mask; skip tooth separation")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=48)
    args = parser.parse_args()
    pipeline = SegmentationPipeline(
        args.weights_dir, device=args.device,
        threshold=args.threshold, stride=args.stride)
    results = pipeline.run(args.cbct, args.output_dir,
                           teeth_mask_path=args.teeth_mask, instances_path=args.instances)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
