from accelerate import Accelerator
from tqdm import tqdm
import torch as tc
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from monai.losses import DiceFocalLoss

def train_tooth_fairy_canals(model, optimizer, train_dataloader, val_dataloader, lr_scheduler, num_epochs, lambda_dice = 1.0, lambda_focal = 1.0, alpha = 0.99):

    CRITERION = DiceFocalLoss(include_background=True, sigmoid=True, lambda_dice=lambda_dice, lambda_focal=lambda_focal, gamma=2, alpha = alpha)

    DEVICE = 'cuda:0'

    accelerator = Accelerator(
        mixed_precision='no',
        gradient_accumulation_steps=1
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    model = model.to(DEVICE)

    global_step = 0
    global_loss = 100

    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), ncols=100, dynamic_ncols=False, leave=False)
        progress_bar.set_description(f"Epoch {epoch}")

        loss_full = 0

        #Training phase
        for step, (clean_images, clean_label) in enumerate(train_dataloader):

            bs = clean_images.shape[0]
            clean_images = clean_images.to(DEVICE)
            clean_label = clean_label.to(DEVICE)

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                label_pred = model(clean_images)

                loss = CRITERION(label_pred, clean_label)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            loss_full += loss.item()

        loss_full = loss_full/len(train_dataloader)
        train_losses.append(loss_full)
        print(f'Epoch: {epoch + 1}, Loss: {loss_full:.7f}')

        loss_val_full = 0
        list_dices = []
        #Validation phase
        with tc.no_grad():
            for step, (clean_images, clean_label) in enumerate(val_dataloader):

                bs = clean_images.shape[0]
                clean_images = clean_images.to(DEVICE)
                clean_label = clean_label.to(DEVICE)

                with accelerator.accumulate(model):
            
                    label_pred = model(clean_images)
                    loss = CRITERION(label_pred, clean_label)
                    dice_score = CRITERION.dice(label_pred, clean_label).item()
                    print(f"Validation Dice Score: {dice_score}")
                    list_dices.append(dice_score)

                loss_val_full += loss.item()

        validation_losses.append(loss_val_full/len(val_dataloader))

        if len(val_dataloader) > 0:
            loss_val_full = loss_val_full/len(val_dataloader)
            list_dices = np.array(list_dices)
            print(f'Epoch: {epoch + 1}, Validation Loss: {loss_val_full:.7f}, Validation DICE: {1 - np.mean(list_dices)}')

        if loss_val_full < global_loss:
            tc.save(model.state_dict(), f'models/weights/tooth_fairy_3d_best.pt')
            print(f"Weights saved! - Epoch {epoch}")
            global_loss = loss_val_full

        tc.cuda.empty_cache()
        print("Cache emptied!")
    
    plt.plot(train_losses, validation_losses)
    plt.legend(['train', 'validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('losses.png')