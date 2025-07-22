from unet3d.pytorch3dunet.unet3d.model import ResidualUNetSE3D

def build_class_segmentor_model(inchannels:int, outchannels:int, depths = [32, 64, 128, 256, 384]):
    
    model = ResidualUNetSE3D(in_channels=inchannels, out_channels=outchannels, final_sigmoid=False, f_maps=depths)

    return model
