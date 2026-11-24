"""Training of the GAN: loss functions and one epoch of training.

TODO: complete `discriminator_loss`, `generator_loss` and `train_one_epoch`.

Notation
    x           : real images
    G(z)        : images generated from latent variables z ~ N(0, I)
    real_logits : outputs of the discriminator for x,    shape (B,)
    fake_logits : outputs of the discriminator for G(z), shape (B,)
    D(.)        : sigmoid(logit), the probability that an image is real
"""

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits, fake_logits):
    """Loss of the discriminator for a batch.

        loss_d = - mean of log D(x)  -  mean of log(1 - D(G(z)))

    It is the binary cross-entropy of the real images with the label 1
    (real) plus the binary cross-entropy of the generated images with the
    label 0 (generated). Each term is averaged over the B images of the
    batch.

    Compute the loss from the logits, not from the probabilities: it is
    numerically stable (see F.binary_cross_entropy_with_logits).

    Args:
        real_logits: outputs of the discriminator for real images,      shape (B,).
        fake_logits: outputs of the discriminator for generated images, shape (B,).

    Returns:
        loss_d: a scalar tensor.
    """
    # TODO
    raise NotImplementedError("mytrain.py: discriminator_loss is not implemented")


def generator_loss(fake_logits):
    """Loss of the generator for a batch (non-saturating loss).

        loss_g = - mean of log D(G(z))

    It is the binary cross-entropy of the generated images with the label 1
    (real), averaged over the B images of the batch: the generator is
    trained to have its images classified as real.

    Args:
        fake_logits: outputs of the discriminator for generated images, shape (B,).

    Returns:
        loss_g: a scalar tensor.
    """
    # TODO
    raise NotImplementedError("mytrain.py: generator_loss is not implemented")


def train_one_epoch(generator, discriminator, optimizer_g, optimizer_d, dataloader, device):
    """Train the GAN for one epoch, i.e. one pass over `dataloader`.

    Put the two networks in training mode first. Then, for every mini-batch
    of B real images x:

        1. Move x to `device`, draw B latent variables z ~ N(0, I) and
           generate the images G(z).
        2. Update the discriminator: compute `discriminator_loss` from its
           outputs for x and for G(z), back-propagate, and update its
           parameters with `optimizer_d`. This loss must not send gradients
           to the generator: detach G(z).
        3. Update the generator: compute `generator_loss` from the outputs
           of the updated discriminator for G(z), back-propagate, and update
           its parameters with `optimizer_g`.

    Args:
        generator:     the generator, already on `device`.
        discriminator: the discriminator, already on `device`.
        optimizer_g:   the optimizer of the generator, from myoptim.get_optimizers.
        optimizer_d:   the optimizer of the discriminator, from myoptim.get_optimizers.
        dataloader:    yields batches of real images, shape (B, 1, 32, 32),
                       values in [0, 1] (there are no labels).
        device:        torch.device to compute on.

    Returns:
        dict {"loss_d": ..., "loss_g": ..., "d_real": ..., "d_fake": ...} of
        Python floats, the average of each term per image over the whole
        epoch:
            loss_d : loss of the discriminator (step 2)
            loss_g : loss of the generator (step 3)
            d_real : D(x), probability given to the real images (step 2)
            d_fake : D(G(z)), probability given to the generated images (step 2)
    """
    # TODO
    raise NotImplementedError("mytrain.py: train_one_epoch is not implemented")
