#Face VAE at Scale
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from google.colab import drive

dataset = ImageFolder(root='./data/celeba/img_align_celeba', transform=transform) # loaded from dataset (downloaded)
dataset = Subset(dataset, range(50000))  # 50k images

batch_size = 64
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(Subset(dataset, range(45000, 50000)), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class FaceVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_ch = 3
        for out_ch in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        n_layers = len(hidden_dims)
        spatial_size = 64 // (2 ** n_layers)
        self.flattened_size = hidden_dims[-1] * (spatial_size ** 2)

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        decoder_layers = []
        rev_dims = list(reversed(hidden_dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(rev_dims[i], rev_dims[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(rev_dims[i+1]),
                nn.ReLU(),
            ])
        decoder_layers.extend([
            nn.ConvTranspose2d(rev_dims[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ])
        self.decoder = nn.Sequential(*decoder_layers)
        self.spatial_size = spatial_size
        self.last_channel = hidden_dims[-1]

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_channel, self.spatial_size, self.spatial_size)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


LATENT_DIM = 128
LEARNING_RATE = 5e-4      # (stable)
EPOCHS = 25
BETA = 0.5

model = FaceVAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

#Training Loop
print("🚀 Starting Training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = total_recon = total_kl = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for x, _ in pbar:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=BETA)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_l.item()
        total_kl += kl_l.item()
        pbar.set_postfix({'loss': f'{loss.item()/len(x):.1f}'})

    # Validation
    model.eval()
    val_loss = val_recon = val_kl = 0
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            loss, r_l, k_l = vae_loss(recon, x, mu, logvar, beta=BETA)
            val_loss += loss.item()
            val_recon += r_l.item()
            val_kl += k_l.item()

    n = len(dataset)
    print(f"Epoch {epoch:2d} | Train Loss: {total_loss/n:.2f} (Recon: {total_recon/n:.2f}, KL: {total_kl/n:.2f})")

    if epoch % 5 == 0 or epoch == EPOCHS:
        torch.save(model.state_dict(), f"/content/drive/MyDrive/face_vae_epoch_{epoch}.pt")

torch.save(model.state_dict(), "/content/drive/MyDrive/face_vae_final.pt")

# Visualizations
@torch.no_grad()
def show_reconstructions(model, loader, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    recon, _, _ = model(x)
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(x[i].permute(1, 2, 0).cpu())
        axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu())
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.suptitle("Original vs Reconstructed")
    plt.show()

show_reconstructions(model, val_loader)

@torch.no_grad()
def generate_faces(model, n=16):
    model.eval()
    z = torch.randn(n, LATENT_DIM).to(device)
    samples = model.decode(z)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].permute(1, 2, 0).cpu())
        ax.axis('off')
    plt.suptitle("Generated Faces")
    plt.show()

generate_faces(model)
