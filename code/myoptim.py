"""Optimization settings: batch size, number of epochs, optimizers."""

import torch

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
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer_g, optimizer_d
