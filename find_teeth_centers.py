from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets.agh_resize_classes import AGH_Resize_Classes
from models.class_segmentor_model import build_class_segmentor_model
from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
from training.center_tooth import train_segmentor
from processing.utils import norm_factory
import argparse

try:
    from clearml import Task
except ImportError:
    Task = None

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
         pretrained = False,
         crop = True,
         freeze = False, 
         num_workers = 0,
         pin_memory = False,
         prefetch_factor = None,
         clearml_task = None): 

    if channels is None:
        channels = []

    elif channels[0] == 'none':
        channels = []

    if prefetch_factor == 0:
        prefetch_factor = None
        
    print(f"Learning parameters: \n Data path: {data_path}\n Batch size: {batch_size}\n Normalization: {normalization}\n Augmentation: {augmentation}\n Learning rate: {learning_rate}\n Weight decay: {weight_decay}\n Number of epochs: {num_epochs}\n Channels: {channels}\n Number of workers: {num_workers}\n Pretrained: {pretrained}\n Freeze: {freeze}")
    norm_func = norm_factory(normalization)
    
    dataset_train = AGH_Resize_Classes(data_directory=data_path,augmentation=augmentation,crop=crop, validation_split=0.15, validation_mode=False)
    dataset_val = AGH_Resize_Classes(data_directory=data_path, augmentation=False, crop=crop, validation_split=0.15, validation_mode=True)

    print(dataset_val.lista)
    
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    inchannels = 1

    model = build_class_segmentor_model(inchannels=inchannels, outchannels=1, weight_path='models/weights/agh_centroid_best_s2.pt', freeze = freeze)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    train_segmentor(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        num_epochs,
        clearml_task=clearml_task,
    )

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_dir', type=str, required=False, default='D:/db1_v1.0')
    arg_parser.add_argument('--num_epochs', type=int, required=False, default=30)
    arg_parser.add_argument('--learning_rate', type=float, required=False, default=1e-3)
    arg_parser.add_argument('--weight_decay', type=float, required=False, default=1e-7)
    arg_parser.add_argument('--batch_size', type=int, required=False, default=2)
    arg_parser.add_argument('--normalization', type=str, required=False, default='none', choices=['standard', '01', 'none'])
    arg_parser.add_argument('--channels', nargs='+', type=str, required=False)
    arg_parser.add_argument('--augmentation', required=False, default=False)
    arg_parser.add_argument('--crop', required=False, default=True)
    arg_parser.add_argument('--pretrained', required=False, default=False)
    arg_parser.add_argument('--freeze', required=False, default=False)
    arg_parser.add_argument('--num_workers', type=int, required=False, default=0)
    arg_parser.add_argument('--pin_memory', required=False, default=False)
    arg_parser.add_argument('--prefetch_factor', required=False, default=None)
    arg_parser.add_argument('--clearml_project', type=str, required=False, default='zeby_tooth')
    arg_parser.add_argument('--clearml_task', type=str, required=False, default='find_teeth_centers')
    arg_parser.add_argument('--clearml_queue', type=str, required=False, default=None)

    args = arg_parser.parse_args()

    if Task is not None and args.clearml_task is not "None":
        
        task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
        task.connect(args)

    if args.clearml_queue:
        task.execute_remotely(queue_name=args.clearml_queue, exit_process=True)

    main(data_path=args.data_dir,
         num_epochs=args.num_epochs,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         batch_size=args.batch_size,
         normalization=args.normalization,
         channels=args.channels,
         augmentation=args.augmentation, 
         pretrained=args.pretrained,
         freeze = args.freeze,
         crop = args.crop,
         num_workers=args.num_workers,
         pin_memory=args.pin_memory,
         prefetch_factor=args.prefetch_factor,
         clearml_task=task,
         )
