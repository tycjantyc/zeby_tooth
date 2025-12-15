from evaluation.evaluate_canals_agh_better import eval
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
import argparse

def main(data_path, file_from, num_files):
    PATH1 = '../../weights_tooth_fairy/tooth_fairy_3d_best_agh.pt'
    FILE = file_from    

    for i in range(num_files):
        
        model = build_hyspark_tooth_fairy_1(in_channel=1, n_classes=1, img_size=96, freeze=False, weight_path=PATH1)
        model.eval()
        model = model.cuda()
        
        eval(model=model, 
            file_cbct=f'{data_path}/db1_v1.0/db1_v1.0_nifti/{FILE}/image.nii.gz', 
            file_instance_seg = f'{data_path}/db1_v1.0/db1_v1.0_nifti_teeth_masks_multi_cls_ordered_weak/{FILE}.nii.gz',
            file_gt=None,
            out = f'../evaluation/eval_results_jenkins/pred_canal_mask_{FILE}.nii.gz')
        
        FILE += 1
    pass

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--file_from', type=int, required=False, default=10)
    arg_parser.add_argument('--num_files', type=float, required=False, default=1e-4)
    
    args = arg_parser.parse_args()

    main(args.data_path, args.file_from, args.num_files)

    