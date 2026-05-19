import torch
import torch.nn.functional as F


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    """
    recon, x : [B, C, H, W] in [-1, 1]
    Returns total loss, recon loss, and KL loss — all scalar tensors.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Averaged over batch and latent dim so it stays on the same scale as recon_loss.
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss
