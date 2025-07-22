import argparse
from torch import load
from evaluation.evaluate import evaluate
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
import numpy as np
import nibabel as nib
from processing.utils import norm_factory

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_weights', type=str, required=False)
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--compression', type=str, required=False, default='medium_rare', choices=['big', 'medium_rare', 'medium', 'none'])
    arg_parser.add_argument('--normalization', type=str, required=False, default='01', choices=['standard', '01', 'none'])
    arg_parser.add_argument('--num_classes', type=int, required=False)
    arg_parser.add_argument('--model_input', type=int, required=False)
    arg_parser.add_argument('--stride', type=int, required=False)
    args = arg_parser.parse_args()

    model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=args.num_classes)
    model.load_state_dict(load(args.model_weights, weights_only=True))
    model.eval()

    stride_shape = (args.stride, args.stride, args.stride)
    input_shape = (args.model_input, args.model_input, args.model_input)

    norm_func = norm_factory(args.normalization)

    segmented_volume = evaluate(model = model, 
                                file = args.file, 
                                model_input_shape = input_shape,
                                stride = stride_shape,
                                num_classes = args.num_classes,  
                                compression_function = args.compression,
                                normalization_function = norm_func)

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(segmented_volume, affine)
    nib.save(nifti_img, 'evaluation/evaluation_results/output_image.nii.gz')