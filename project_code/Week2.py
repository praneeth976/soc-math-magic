# Submission of same has also be done


import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

def kl_divergence_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence between N(mu, exp(logvar)) and N(0, 1).
    Returns a scalar (summed over all dimensions and batch).
    """
    # My Code
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Test it
mu     = torch.randn(32, 16)   # batch of 32, latent dim 16
logvar = torch.randn(32, 16)
sigma  = torch.exp(0.5 * logvar)

your_kl = kl_divergence_gaussian(mu, logvar)

q = Normal(mu, sigma)
p = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
lib_kl  = kl_divergence(q, p).sum()

print(f"Your KL:   {your_kl:.4f}")
print(f"Torch KL:  {lib_kl:.4f}")
assert torch.isclose(your_kl, lib_kl, atol=1e-4), "KL values don't match!"
print(" Part 1 passed")

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Sample z using the reparameterization trick.
    z = mu + std * eps, where eps ~ N(0, I)
    """
    # My Code
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

# Verify: the mean of many samples should be close to mu
mu_test     = torch.tensor([2.0, -1.0])
logvar_test = torch.tensor([0.0,  0.5])

samples = torch.stack([reparameterize(mu_test, logvar_test) for _ in range(10000)])
print(f"Target mu:       {mu_test.tolist()}")
print(f"Sample mean:     {samples.mean(0).tolist()}")
print(f"Target std:      {torch.exp(0.5 * logvar_test).tolist()}")
print(f"Sample std:      {samples.std(0).tolist()}")
# Means and stds should be close (within ~0.05)
print("Part 2 passed")

def elbo_loss(x: torch.Tensor,
              x_recon: torch.Tensor,
              mu: torch.Tensor,
              logvar: torch.Tensor) -> torch.Tensor:
    """
    ELBO loss = Reconstruction loss + KL divergence
    - Reconstruction: BCE between x_recon and x, summed over pixels and batch
    - KL: closed-form KL between N(mu, exp(logvar)) and N(0, I)
    Returns a scalar.
    """
    # My Code
    reconstruction_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss+kl_loss

# Quick smoke test
batch, dim, latent = 16, 784, 32
x      = torch.rand(batch, dim)
x_recon = torch.sigmoid(torch.randn(batch, dim))
mu     = torch.randn(batch, latent)
logvar = torch.randn(batch, latent)

loss = elbo_loss(x, x_recon, mu, logvar)
print(f"ELBO loss (should be a positive scalar): {loss.item():.4f}")
assert loss.ndim == 0, "Loss must be a scalar!"
print("Part 3 passed")
