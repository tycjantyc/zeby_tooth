import numpy as np
from processing.utils import create_random_snippet

def prepare_toothfairy(image: np.ndarray, 
                       mask: np.ndarray, 
                       remove_ct_rings:bool, 
                       input_size: tuple[int, int, int],  
                       normalization_function, 
                       augmentation:bool, 
                       channels:list,
                       patchify:bool = True,
                       coords:bool = False):
    
        image = image.astype(np.float32)

        assert image.shape == mask.shape

        if patchify:
            image, mask, coordinations = create_random_snippet(image, mask, input_size, coords=True)

        image = normalization_function(image)

        if augmentation:
            from processing.data_augmentation import AugmentationFactory
            aug_factory = AugmentationFactory()
            image, mask = aug_factory.create(image, mask)

        if channels:
            from processing.multi_input import MultiInputFactory
            image = MultiInputFactory(channels).create(image)
        else:
            image = image[None, ...]

        if coords:
            return image, mask, coordinations
        else:
            return image, mask
