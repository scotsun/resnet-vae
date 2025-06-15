import torch
import torch.nn.functional as F


def sample_level_reduction(tensor: torch.Tensor):
    n_dims = len(tensor.shape)
    return tensor.sum(dim=list(range(n_dims))[1:]).mean()


def vae_loss(x_reconstr, x, mu, logvar):
    """
    VAE loss with separating factors.
    """
    reconstruction_loss = sample_level_reduction(
        F.mse_loss(x_reconstr, x, reduction="none")
    )

    kl = -0.5 * sample_level_reduction(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss, kl
