import numpy as np
from processing.utils import create_random_snippet
from processing.data_augmentation import AugmentationFactory
from processing.multi_input import MultiInputFactory

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
            aug_factory = AugmentationFactory()
            image, mask = aug_factory.create(image, mask)

        multi_input = MultiInputFactory(channels)
        image = multi_input.create(image)

        if coords:
            return image, mask, coordinations
        else:
            return image, mask