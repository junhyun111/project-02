"""Training of the GAN: loss functions and one epoch of training.

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
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real_loss + fake_loss


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
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


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
    generator.train()
    discriminator.train()

    generator_parameters = tuple(generator.parameters())
    metric_names = ("loss_d", "loss_g", "d_real", "d_fake")
    totals = torch.zeros(len(metric_names), device=device)
    num_images = 0

    for images in dataloader:
        images = images.to(device)
        batch_size = images.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake = generator(z)

        optimizer_d.zero_grad()
        real_logits = discriminator(images)
        fake_logits = discriminator(fake.detach())
        loss_d = discriminator_loss(real_logits, fake_logits)
        loss_d.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        loss_g = generator_loss(discriminator(fake))
        loss_g.backward(inputs=generator_parameters)
        optimizer_g.step()

        with torch.no_grad():
            batch_metrics = torch.stack((
                loss_d,
                loss_g,
                real_logits.sigmoid().mean(),
                fake_logits.sigmoid().mean(),
            ))
            totals += batch_size * batch_metrics
        num_images += batch_size

    if num_images == 0:
        raise ValueError("train_one_epoch requires at least one training image")
    return dict(zip(metric_names, (totals / num_images).tolist()))
