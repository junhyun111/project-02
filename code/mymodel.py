"""Generative adversarial network (GAN): generator and discriminator.

TODO: complete the classes `Generator` and `Discriminator`.

You are free to design the architectures (type and number of layers, their
sizes, the latent dimension), but keep the interface described in the
docstrings: the notebook and the other files rely on it.

Notation
    B      : number of images in a batch
    z      : latent variables,             shape (B, latent_dim), z ~ N(0, I)
    x      : real images,                  shape (B, 1, 32, 32), values in [0, 1]
    G(z)   : generated images,             shape (B, 1, 32, 32), values in [0, 1]
    logits : outputs of the discriminator, shape (B,), one real number per image
    D(x)   : sigmoid(logit), the probability that the image x is real
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        """Define the layers of the generator.

        The notebook creates the generator with `Generator()`, so every
        argument you add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.
        """
        super().__init__()
        # TODO: set self.latent_dim and define the layers
        raise NotImplementedError("mymodel.py: Generator.__init__ is not implemented")

    def forward(self, z):
        """Generator: images G(z) generated from the latent variables z.

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            generated images, shape (B, 1, 32, 32), values in [0, 1].
        """
        # TODO
        raise NotImplementedError("mymodel.py: Generator.forward is not implemented")


class Discriminator(nn.Module):
    def __init__(self):
        """Define the layers of the discriminator.

        The notebook creates the discriminator with `Discriminator()`, so
        every argument you add needs a default value.
        """
        super().__init__()
        # TODO: define the layers
        raise NotImplementedError("mymodel.py: Discriminator.__init__ is not implemented")

    def forward(self, x):
        """Discriminator: one logit per image.

        The logit is a real number: do not apply a sigmoid at the end. The
        probability that the image x is real is D(x) = sigmoid(logit).

        Args:
            x: real or generated images, shape (B, 1, 32, 32), values in [0, 1].

        Returns:
            logits, shape (B,).
        """
        # TODO
        raise NotImplementedError("mymodel.py: Discriminator.forward is not implemented")
