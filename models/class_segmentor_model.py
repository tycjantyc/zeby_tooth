from unet3d.pytorch3dunet.unet3d.model import ResidualUNetSE3D
from torch import load

def build_class_segmentor_model(inchannels:int, outchannels:int, depths = [32, 64, 128, 256, 384], weight_path = None):
    
    model = ResidualUNetSE3D(in_channels=inchannels, out_channels=outchannels, final_sigmoid=False, f_maps=depths, is_segmentation=False)
    
    if weight_path is not None:
        model.load_state_dict(load(weight_path, weights_only=True))
        print("Weights loaded!")

    return model
