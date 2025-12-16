from evaluation.evaluate_canals_agh_better import eval
from models.class_segmentor_model import build_class_segmentor_model
from evaluation.test import instance_teeth_segment
from data_prepare.watershed_test_2_0 import main as main_watershed
import argparse
from torch import load
from tqdm import tqdm

def main(data_path, file_from, num_files):

    FILE = file_from    
    model = build_class_segmentor_model(inchannels=1, outchannels=34, depths=[32, 64, 128, 256, 384])
    model.load_state_dict(load('models/weights/agh_segmentor_best_2.pt', weights_only=True))
    model.eval()
    model = model.cuda()

    for i in tqdm(range(num_files)):

        i = i+FILE

        file_semantic = f'{data_path}/db1_v1.0/db1_v1.0_nifti_teeth_masks_single_cls_weak/{i}.nii.gz'
        file_instance = f'{data_path}/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{i}.nii.gz'
        helper = f'evaluation/eval_results_jenkins/pred_teeth_helper_mask_{i}.nii.gz'
        out = f'evaluation/eval_results_jenkins/pred_teeth_mask_{i}.nii.gz'

        instance_teeth_segment(model, file_semantic, file_instance, helper)

        main_watershed(seed_mask=helper, ws_mask=file_semantic, out=out)

        FILE += 1
    pass

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--file_from', type=int, required=False, default=1)
    arg_parser.add_argument('--num_files', type=int, required=False, default=10)
    
    args = arg_parser.parse_args()

    main(args.data_path, args.file_from, args.num_files)

    