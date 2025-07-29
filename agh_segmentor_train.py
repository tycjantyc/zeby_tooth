from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.agh_class_segmentor import AGH_Class_Segmentor_Dataset
from datasets.agh_resize_classes import AGH_Resize_Classes
from models.class_segmentor_model import build_class_segmentor_model
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from training.train_segmentor_agh import train_segmentor_agh
from processing.utils import norm_factory
import argparse

def main(data_path = 'D:/db1_v1.0', 
         batch_size = 1, 
         min_clamp = -1000, 
         max_clamp = 3000, 
         normalization = '01', #'standard', '01', 'none'
         augmentation = False, 
         learning_rate = 1e-3, 
         weight_decay = 1e-6, 
         num_epochs = 10,
         lr_warmup_steps = 20,
         channels = [],  # Choose from: 'sobel', 'clahe', 'median'
         num_workers = 0,
         pin_memory = False,
         prefetch_factor = None): 

    if channels is None:
        channels = []

    elif channels[0] == 'none':
        channels = []

    if prefetch_factor == 0:
        prefetch_factor = None
        
    print(f"Learning parameters: \n Data path: {data_path}\n Batch size: {batch_size}\n Normalization: {normalization}\n Augmentation: {augmentation}\n Learning rate: {learning_rate}\n Weight decay: {weight_decay}\n Number of epochs: {num_epochs}\n Channels: {channels}\n Number of workers: {num_workers}")

    norm_func = norm_factory(normalization)

    # dataset_train = AGH_Class_Segmentor_Dataset(data_directory=data_path, clamp = (min_clamp, max_clamp), normalization=norm_func,augmentation=augmentation, channels=channels, validation_split=0.05, validation_mode=False)
    # dataset_val = AGH_Class_Segmentor_Dataset(data_directory=data_path, clamp = (min_clamp, max_clamp), normalization=norm_func,augmentation=False, channels=channels, validation_split=0.05, validation_mode=True)
    
    dataset_train = AGH_Resize_Classes(data_directory=data_path,augmentation=augmentation, validation_split=0.05, validation_mode=False)
    dataset_val = AGH_Resize_Classes(data_directory=data_path, augmentation=False, validation_split=0.05, validation_mode=True)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    inchannels = 1

    model = build_class_segmentor_model(inchannels=inchannels, outchannels=34, depths=[16, 32, 64, 128, 192])
    #model = build_hyspark_tooth_fairy_1(in_channel=inchannels, n_classes=2, freeze=False)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_segmentor_agh(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=False, default='D:/db1_v1.0')
    arg_parser.add_argument('--num_epochs', type=int, required=False, default=10)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=1e-4)
    arg_parser.add_argument('--weight_decay', type=float, required=False, default=1e-6)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=1)
    arg_parser.add_argument('--normalization', type=str, required=False, default='none', choices=['standard', '01', 'none'])
    arg_parser.add_argument('--channels', nargs='+', type=str, required=False)
    arg_parser.add_argument('--augmentation', required=False, default=False)
    arg_parser.add_argument('--num_workers', type=int, required=False, default=0)
    arg_parser.add_argument('--pin_memory', required=False, default=False)
    arg_parser.add_argument('--prefetch_factor', required=False, default=None)

    args = arg_parser.parse_args()

    main(data_path=args.data_dir,
         num_epochs=args.num_epochs,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         normalization=args.normalization,
         channels=args.channels,
         augmentation=args.augmentation, 
         num_workers=args.num_workers,
         pin_memory=args.pin_memory,
         prefetch_factor=args.prefetch_factor
         )


