"""Generation of new images with a trained generator.

TODO: complete `sample`.
"""

import torch


def sample(generator, num_samples, device):
    """Generate new images: draw z from N(0, I) and apply the generator.

    The generator is not updated: put it in evaluation mode and do not track
    gradients (torch.no_grad).

    Args:
        generator:   the generator, already on `device`.
        num_samples: number of images to generate.
        device:      torch.device to compute on.

    Returns:
        generated images, shape (num_samples, 1, 32, 32), values in [0, 1].
    """
    # TODO
    raise NotImplementedError("mysample.py: sample is not implemented")
