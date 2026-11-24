"""Result of a trained GAN (provided: do not modify this file).

Two functions are used by the notebook:

    evaluate(generator)
        measures the trained generator with the testing data, the 5,000
        images of `get_dataset("test")` in data.py:
            FID (Frechet inception distance) between 5,000 images generated
                by the generator and the 5,000 test images
        Lower is better.

    save(result, info)
        saves the value in the file
            result/result_<student ID>_<name>.csv

Usage (with the `util` and `code` folders on sys.path):
    from myinfo import get_info
    from result import evaluate, save

    result = evaluate(generator)        # generator: the trained generator
    save(result, get_info())

The FID is computed with the package `torcheval`:
    pip install torcheval torchvision
It uses the Inception-v3 network of torchvision, pretrained on ImageNet. Its
weights (about 100 MB) are downloaded the first time `evaluate` is called,
which requires an internet connection, and are then read from the cache
folder of PyTorch. `evaluate` takes seconds with a GPU, but can take more
than 10 minutes with a CPU.
"""

import csv
from pathlib import Path

import torch
from torcheval.metrics import FrechetInceptionDistance

from data import get_dataset

# <project root>/result, independent of the directory the notebook is run from
RESULT_DIR = Path(__file__).resolve().parent.parent / "result"

# Number of images processed at a time. It is kept small because the
# Inception-v3 network works on images enlarged to 299 x 299 pixels.
BATCH_SIZE = 50

# Seed of the latent variables of the generated images: the same latent
# variables are used every time, so that the FID of a generator does not
# change from one evaluation to the next.
SEED = 0


@torch.no_grad()
def evaluate(generator):
    """Measure a trained generator with the test images.

    FID: as many images as test images (5,000) are generated from latent
    variables z drawn from N(0, I). The FID is the Frechet distance between
    the Gaussian distributions fitted to the Inception-v3 features of the
    generated images and of the test images. It is 0 when the two Gaussians
    are the same.

    The generator is not updated: it is put in evaluation mode and gradients
    are not tracked. The computation runs on the device of the generator.
    The discriminator is not used.

    Args:
        generator: the trained generator, with
                       latent_dim         dimension of z
                       generator(z) -> x  z of shape (B, latent_dim),
                                          x of shape (B, 1, 32, 32), values in [0, 1]

    Returns:
        dict {"fid": ...} of a Python float.
    """
    device = next(generator.parameters()).device
    test_images = get_dataset("test").images
    generator.eval()

    # latent variables of the generated images, one per test image
    rng = torch.Generator().manual_seed(SEED)
    z = torch.randn(len(test_images), generator.latent_dim, generator=rng)

    fid = FrechetInceptionDistance(device=device)

    for x, z_batch in zip(test_images.split(BATCH_SIZE), z.split(BATCH_SIZE)):
        x = x.to(device)

        # generated images
        samples = generator(z_batch.to(device))
        if samples.shape != x.shape:
            raise ValueError(
                f"evaluate: the generator returns images of shape {tuple(samples.shape)}, "
                f"expected {tuple(x.shape)}"
            )

        fid.update(_to_rgb(x), is_real=True)
        fid.update(_to_rgb(samples), is_real=False)

    return {"fid": fid.compute().item()}


def save(result, info):
    """Save the result of `evaluate` in a CSV file.

    The file is result/result_<student ID>_<name>.csv. It has a header line
    and one line of values:

        student_id,name,fid

    Args:
        result: the dict returned by `evaluate`.
        info:   the dict returned by `get_info` in myinfo.py:
                {"name": ..., "student_id": ...}.

    Returns:
        the path of the saved file.
    """
    name = str(info["name"]).strip()
    student_id = str(info["student_id"]).strip()
    if not name or not student_id:
        raise ValueError("save: fill in your name and student ID in myinfo.py")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"result_{_for_filename(student_id)}_{_for_filename(name)}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id", "name", "fid"])
        writer.writerow([student_id, name, result["fid"]])

    return path


def _to_rgb(images):
    """Gray images (B, 1, H, W) -> RGB images (B, 3, H, W), float32 with
    values in [0, 1], as expected by the FID of torcheval."""
    return images.float().clamp(0, 1).repeat(1, 3, 1, 1)


def _for_filename(text):
    """Replace the characters that are not letters or digits (spaces, /, ...)
    by "_", so that the text can be part of a file name."""
    return "".join(c if c.isalnum() else "_" for c in text)
