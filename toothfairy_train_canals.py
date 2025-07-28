from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.tooth_fairy_3_canals_dataset import ToothFairy3_Canal_Dataset
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from models.class_segmentor_model import build_class_segmentor_model
from training.train_toothfairy_canals import train_tooth_fairy_canals
from processing.tooth_fairy_classes import compression_factory
from processing.utils import norm_factory
import argparse
from torch import load
import os

def main(data_path = 'data', 
         batch_size = 1, 
         augmentation = True, 
         learning_rate = 1e-3, 
         weight_decay = 1e-6, 
         num_epochs = 10,
         lr_warmup_steps = 200,
         inputsize = 96,
         model = 'hyspark',
         channels = [],  # Choose from: 'sobel', 'clahe', 'median'
         validation_split = 0.05,
         weights_file = 'models/weights/tooth_fairy_3d_best.pt',
         lambda_dice = 1,
         lambda_focal = 1,
         alpha_focal = 0.9,
         num_workers = 0,
         pin_memory = False,
         prefetch_factor = None): 

    if channels is None:
        channels = []

    elif channels[0] == 'none':
        channels = []

    if prefetch_factor == 0:
        prefetch_factor = None
        
    print(f"Learning parameters: \n Data path: {data_path}\n Batch size: {batch_size}\n Augmentation: {augmentation}\n Learning rate: {learning_rate}\n Weight decay: {weight_decay}\n Number of epochs: {num_epochs}\n Channels: {channels}\n Number of workers: {num_workers}")

    dataset_train = ToothFairy3_Canal_Dataset(data_directory=data_path, input_size=(inputsize, inputsize, inputsize), augmentation=augmentation, validation_split=validation_split)
    dataset_val = ToothFairy3_Canal_Dataset(data_directory=data_path, input_size=(inputsize, inputsize, inputsize), augmentation=False, validation_split=validation_split, validation_mode=True)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, prefetch_factor=None)

    inchannels = 1 + len(channels)

    if model == 'hyspark':
        model = build_hyspark_tooth_fairy_1(in_channel=inchannels, n_classes=1, img_size=inputsize, freeze=False)
    elif model == 'resunet3dse':
        model = build_class_segmentor_model(inchannels=inchannels, outchannels=1)
    
    if os.path.isfile(weights_file):
        model.load_state_dict(load(weights_file, weights_only=True))
        print('Weights loaded!')

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_tooth_fairy_canals(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs, lambda_dice, lambda_focal, alpha_focal)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--num_epochs', type=int, required=False, default=10)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=1e-4)
    arg_parser.add_argument('--weight_decay', type=float, required=False, default=1e-6)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=1)
    arg_parser.add_argument('--input_size', type=int, required=False, default=96)
    arg_parser.add_argument('--channels', nargs='+', type=str, required=False)
    arg_parser.add_argument('--augmentation', required=False, default=True)
    arg_parser.add_argument('--model', choices=['hyspark', 'resunet3dse'])
    arg_parser.add_argument('--lambda_dice', type=float, required=False, default=1.0)
    arg_parser.add_argument('--lambda_focal', type=float, required=False, default=1.0)
    arg_parser.add_argument('--alpha', type=float, required=False, default=0.99)
    arg_parser.add_argument('--num_workers', type=int, required=False, default=0)
    arg_parser.add_argument('--pin_memory', required=False, default=False)
    arg_parser.add_argument('--prefetch_factor', required=False, default=None)

    args = arg_parser.parse_args()

    main(data_path=args.data_dir,
         num_epochs=args.num_epochs,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         inputsize=args.input_size,
         channels=args.channels,
         model = args.model,
         augmentation=args.augmentation, 
         lambda_dice=args.lambda_dice,
         lambda_focal=args.lambda_focal,
         alpha_focal=args.alpha,
         num_workers=args.num_workers,
         pin_memory=args.pin_memory,
         prefetch_factor=args.prefetch_factor
         )


