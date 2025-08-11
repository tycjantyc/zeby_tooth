import torchio as tio
import numpy as np
import torch

class AugmentationFactory():
    def __init__(self):
        self.transform = tio.Compose([
                                        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                                        tio.RandomAffine(
                                            scales=(0.9, 1.2),
                                            degrees=15,
                                        )
                                        #tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3)
                                    ])

    def create(self, img_np, mask_np):

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(img_np[None, ...])),
            mask=tio.LabelMap(tensor=torch.from_numpy(mask_np[None,...]))
        )

        transformed = self.transform(subject)
        img_aug = transformed.image.numpy().squeeze().astype(np.float32)
        mask_aug = transformed.mask.numpy().squeeze().astype(np.float32)

        return img_aug, mask_aug
