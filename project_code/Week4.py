# pretty much the same as week 3 but convulutional VAE are used which is better for images
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =Model
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20, hidden_dims=None):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.hidden_dims = hidden_dims

        # Encoder
        encoder_layers = []
        in_ch = 1
        for i, out_ch in enumerate(hidden_dims):
            pad = 0 if i == len(hidden_dims)-1 else 1
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=pad),
                nn.ReLU()
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        self.flattened_size = hidden_dims[-1] * 3 * 3
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        decoder_layers = []
        rev_dims = list(reversed(hidden_dims))
        for i in range(len(rev_dims)-1):
            decoder_layers.extend([
                nn.ConvTranspose2d(rev_dims[i], rev_dims[i+1], kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            ])
        decoder_layers.extend([
            nn.ConvTranspose2d(rev_dims[-1], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.hidden_dims[-1], 3, 3)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

#Loss
def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_resized = F.interpolate(recon, size=(28, 28), mode='bilinear')
    recon_loss = F.binary_cross_entropy(recon_resized, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld

#Training
def train(model, loader, optimizer, epochs=30, beta=1.0):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader.dataset):.2f}")

# ==================== Visualization ===================
def plot_reconstructions(model, data_loader, n=10):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(data_loader))
        x = x[:n].to(device)
        recon, _, _ = model(x)
        recon = F.interpolate(recon, size=(28, 28), mode='bilinear')

    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.suptitle('Top: Original | Bottom: Reconstructed')
    plt.show()

def plot_generated(model, n=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim).to(device)
        samples = model.decode(z)
        samples = F.interpolate(samples, size=(28, 28), mode='bilinear')

    fig, axes = plt.subplots(1, n, figsize=(15, 2))
    for i in range(n):
        axes[i].imshow(samples[i].cpu().squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle('Generated Samples')
    plt.show()

# Execution
if __name__ == "__main__":
    model = ConvVAE(latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training ConvVAE on MNIST...")
    train(model, train_loader, optimizer, epochs=30, beta=1.0)

    print("\nVisualizing results...")
    plot_reconstructions(model, test_loader)
    plot_generated(model)
