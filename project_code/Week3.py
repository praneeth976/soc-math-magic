# VAE is an intelligent way for generative Neural Networks
# the code is the succession of the math discussed before
# Two Neural Networks are trained namely encoder , decoder 
# the whole architecture is inbuilt in the module nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
import numpy as np

# Data
def get_moons_data(n_samples=3000, batch_size=256):
    X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    X = torch.FloatTensor(X)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, X

# Model
class LinearVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# Loss
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# Training
def train(model, loader, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = total_recon = total_kl = 0
        for batch in loader:
            x = batch[0]
            recon, mu, logvar = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()

        if epoch % 20 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch:3d} | Total: {total_loss:.1f} | Recon: {total_recon:.1f} | KL: {total_kl:.1f}")

#Visualze
def plot_reconstructions(model, data, n=10):
    with torch.no_grad():
        recon, _, _ = model(data[:n])
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:n, 0], data[:n, 1], c='blue', alpha=0.7, label='Original')
    plt.title('Original Data')
    plt.subplot(1, 2, 2)
    plt.scatter(recon[:, 0], recon[:, 1], c='red', alpha=0.7, label='Reconstructed')
    plt.title('Reconstructed Data')
    plt.legend()
    plt.show()

def plot_latent_space(model, data):
    with torch.no_grad():
        mu, _ = model.encode(data)
    plt.figure(figsize=(6, 6))
    plt.scatter(mu[:, 0], mu[:, 1], alpha=0.6, s=15)
    plt.title('Latent Space')
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_generated(model, n=1000):
    with torch.no_grad():
        z = torch.randn(n, model.decoder.in_features)  # latent_dim
        samples = model.decode(z)
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=8)
    plt.title('Generated Samples (from N(0, I))')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    loader, data = get_moons_data()
    model = LinearVAE(input_dim=2, latent_dim=2)
    
    print("Training Linear VAE...")
    train(model, loader, epochs=200)
    
    print("\nGenerating plots...")
    plot_reconstructions(model, data)
    plot_latent_space(model, data)
    plot_generated(model)
