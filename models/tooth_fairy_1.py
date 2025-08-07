from hyspark.models.network.hyspark_model import build_hybird, UnetOutBlock, Hybird
import torch
import torch.nn as nn
import os

def build_hyspark_tooth_fairy_1(in_channel = 1, n_classes = 77, img_size = 96, freeze = True, weight_path = None):

    model = Hybird(in_channel=in_channel,n_classes = n_classes, channels = (64, 128, 256, 384, 512))
    
    if in_channel != 1:
        model.embeddings.stem = nn.Conv3d(in_channels=in_channel, out_channels=model.embeddings.channels[0], kernel_size=3, stride=1, padding=1)
        for param in model.embeddings.stem.parameters():
            param.requires_grad = True

    if freeze:
        for param1 in model.embeddings.parameters():
            param1.requires_grad = False

        for param2 in model.mae.parameters():
            param2.requires_grad = False

    model.decoder.out = UnetOutBlock(in_channels=model.decoder.out.in_channels, n_classes=n_classes)
    
    if weight_path is not None and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        print("Weights loaded!")

    return model
