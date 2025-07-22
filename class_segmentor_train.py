from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.tooth_fairy_class_segmentor import ToothFairy3_Class_Segmentor_Dataset
from models.class_segmentor_model import build_class_segmentor_model
from training.train_segmentor_tooth_fair import train_class_segmentor
from processing.tooth_fairy_classes import compression_factory
from processing.utils import norm_factory
import argparse

def main(data_path = 'D:/ToothFairy3', 
         batch_size = 1, 
         min_clamp = -1000, 
         max_clamp = 3000, 
         compression = 'medium',  #'big' - 3 classes, 'medium' - 8 classes, 'none' - 77 classes
         normalization = '01', #'standard', '01', 'none'
         augmentation = True, 
         learning_rate = 1e-3, 
         weight_decay = 1e-6, 
         num_epochs = 10,
         lr_warmup_steps = 200,
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
        
    print(f"Learning parameters: \n Data path: {data_path}\n Batch size: {batch_size}\n Compression: {compression}\n Normalization: {normalization}\n Augmentation: {augmentation}\n Learning rate: {learning_rate}\n Weight decay: {weight_decay}\n Number of epochs: {num_epochs}\n Freeze pretrainied layers: {freeze}\n Channels: {channels}\n Number of workers: {num_workers}")

    comp_func, num_classes = compression_factory(compression)
    norm_func = norm_factory(normalization)

    dataset_train = ToothFairy3_Class_Segmentor_Dataset(data_directory=data_path, clamp = (min_clamp, max_clamp), compression_function = comp_func, normalization=norm_func,augmentation=augmentation, channels=channels)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

    inchannels = 1 + len(channels)

    model = build_class_segmentor_model(in_channel=inchannels, n_classes=num_classes, depths=[16, 32, 64, 128, 192])
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_class_segmentor(model, optimizer, train_dataloader, lr_scheduler, num_epochs, num_classes)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=False, default='D:/ToothFairy3')
    arg_parser.add_argument('--num_epochs', type=int, required=False, default=10)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=1e-4)
    arg_parser.add_argument('--weight_decay', type=float, required=False, default=1e-6)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=1)
    arg_parser.add_argument('--compression', type=str, required=False, default='medium_rare', choices=['big', 'medium_rare', 'medium', 'none'])
    arg_parser.add_argument('--normalization', type=str, required=False, default='01', choices=['standard', '01', 'none'])
    arg_parser.add_argument('--channels', nargs='+', type=str, required=False)
    arg_parser.add_argument('--augmentation', required=False, default=True)
    arg_parser.add_argument('--num_workers', type=int, required=False, default=0)
    arg_parser.add_argument('--pin_memory', required=False, default=False)
    arg_parser.add_argument('--prefetch_factor', required=False, default=None)

    args = arg_parser.parse_args()

    main(data_path=args.data_dir,
         num_epochs=args.num_epochs,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         compression=args.compression,
         normalization=args.normalization,
         channels=args.channels,
         augmentation=args.augmentation, 
         num_workers=args.num_workers,
         pin_memory=args.pin_memory,
         prefetch_factor=args.prefetch_factor
         )


