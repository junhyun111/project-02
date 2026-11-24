"""Generative adversarial network (GAN): generator and discriminator.

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

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        """Define the layers of the generator.

        The notebook creates the generator with `Generator()`, so every
        argument you add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.
        """
        super().__init__()
        self.latent_dim = 64
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * 4 * 4, bias=False),
            nn.Unflatten(1, (128, 4, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """Generator: images G(z) generated from the latent variables z.

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            generated images, shape (B, 1, 32, 32), values in [0, 1].
        """
        return self.network(z)


class Discriminator(nn.Module):
    def __init__(self):
        """Define the layers of the discriminator.

        The notebook creates the discriminator with `Discriminator()`, so
        every argument you add needs a default value.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, x):
        """Discriminator: one logit per image.

        The logit is a real number: do not apply a sigmoid at the end. The
        probability that the image x is real is D(x) = sigmoid(logit).

        Args:
            x: real or generated images, shape (B, 1, 32, 32), values in [0, 1].

        Returns:
            logits, shape (B,).
        """
        return self.network(x).squeeze(-1)
