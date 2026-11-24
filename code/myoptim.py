"""Optimization settings: batch size, number of epochs, optimizers.

TODO: choose BATCH_SIZE and NUM_EPOCHS, and complete `get_optimizers`.
"""

import torch

# Used by the notebook. The values below are only a starting point.
BATCH_SIZE = 64     # number of images in a mini-batch
NUM_EPOCHS = 400    # number of passes over the training data


def get_optimizers(generator, discriminator):
    """Create the two optimizers: one updates the parameters of `generator`,
    the other one the parameters of `discriminator`.

    The choice of the optimizers and of their settings (learning rate, ...)
    is yours.

    Args:
        generator:     the generator defined in mymodel.py.
        discriminator: the discriminator defined in mymodel.py.

    Returns:
        (optimizer_g, optimizer_d): two torch.optim.Optimizer, for the
        generator and for the discriminator.
    """
    # TODO
    raise NotImplementedError("myoptim.py: get_optimizers is not implemented")
