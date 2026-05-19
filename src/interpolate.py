import torch
import numpy as np

from src.model import VAE


def interpolate(model: VAE, img_a: torch.Tensor, img_b: torch.Tensor, steps: int, device) -> list[torch.Tensor]:
    """
    img_a, img_b : [1, 3, H, W] normalised to [-1, 1]
    Returns a list of `steps` decoded images (including endpoints), each [1, 3, H, W].
    """
    model.eval()
    with torch.no_grad():
        z_a = model.encode(img_a.to(device))
        z_b = model.encode(img_b.to(device))

        ts = np.linspace(0, 1, steps)
        frames = []
        for t in ts:
            z = (1 - t) * z_a + t * z_b
            frame = model.decode(z)
            frames.append(frame.cpu())
    return frames


def generate(model: VAE, n: int, device) -> torch.Tensor:
    """Sample n random faces from N(0, I). Returns [n, 3, H, W]."""
    model.eval()
    with torch.no_grad():
        return model.sample(n, device).cpu()
