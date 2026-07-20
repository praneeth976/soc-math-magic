!pip install imageio tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FaceVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=None, kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Encoder
        encoder_layers = []
        in_ch = 3
        for out_ch in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

        n_layers = len(hidden_dims)
        spatial_size = 64 // (2 ** n_layers)
        self.flattened_size = hidden_dims[-1] * (spatial_size ** 2)
        self.spatial_size = spatial_size
        self.last_channel = hidden_dims[-1]

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

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_channel, self.spatial_size, self.spatial_size)
        return self.decoder(h)

model = FaceVAE(latent_dim=128).to(device)
print("Model architecture ready!")

#get the model from the drive if saved

image_size = 64
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
dataset_with_attrs = datasets.CelebA(root="./data", split="train", download=True,
                                     transform=transform, target_type='attr')

loader = DataLoader(dataset, batch_size=64, shuffle=True)
attr_loader = DataLoader(dataset_with_attrs, batch_size=128, shuffle=True)

print(f"Dataset loaded: {len(dataset)} images")

@torch.no_grad()
def face_morph(model, img1, img2, steps=30):
    """Create smooth morph between two faces"""
    mu1, _ = model.encode(img1.to(device))
    mu2, _ = model.encode(img2.to(device))

    frames = []
    for alpha in np.linspace(0, 1, steps):
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        recon = model.decode(z_interp)
        frame = recon[0].cpu().permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        frames.append((frame * 255).astype(np.uint8))

    return frames

# ================== Generate Morph ==================
x, _ = next(iter(loader))   # Get a batch of images

# Choose two different faces (you can change indices)
img1 = x[0:1]   # First face
img2 = x[1:2]   # Second face

print("Generating morphing GIF...")

frames = face_morph(model, img1, img2, steps=30)

# Save GIF
imageio.mimsave('face_morph.gif', frames, duration=0.1, loop=0)
print("✅ GIF saved as 'face_morph.gif' !")

# Show preview (Start, Middle, End)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(frames[0])
axes[0].set_title("Start Face")
axes[0].axis('off')

axes[1].imshow(frames[15])
axes[1].set_title("Middle (50%)")
axes[1].axis('off')

axes[2].imshow(frames[-1])
axes[2].set_title("End Face")
axes[2].axis('off')

plt.suptitle("Face Morphing in Latent Space", fontsize=16)
plt.tight_layout()
plt.show()



ATTR_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

@torch.no_grad()
def encode_batch(model, loader, max_batch=8000):
    all_mu = []
    all_attrs = None
    collected = 0
    for x, attrs in tqdm(loader, desc="Encoding"):
        mu, _ = model.encode(x.to(device))
        all_mu.append(mu.cpu())
        if all_attrs is None:
            all_attrs = attrs
        else:
            all_attrs = torch.cat([all_attrs, attrs])
        collected += x.shape[0]
        if collected >= max_batch:
            break
    return torch.cat(all_mu), all_attrs

def get_attribute_vector(model, loader_with_attrs, attr_name, n_samples=8000):
    attr_idx = ATTR_NAMES.index(attr_name)
    mu_vectors, attr_labels = encode_batch(model, loader_with_attrs, max_batch=n_samples)

    # Robust label handling (handles both -1/1 and 0/1)
    labels = attr_labels[:, attr_idx]
    has_attr = labels == 1
    no_attr = (labels == -1) | (labels == 0)   # ← This fixes it!

    mu_with = mu_vectors[has_attr]
    mu_without = mu_vectors[no_attr]

    print(f"{attr_name}: {len(mu_with)} positive, {len(mu_without)} negative")

    if len(mu_with) < 100 or len(mu_without) < 100:
        print(f"⚠️ Low samples for {attr_name}. Using what we have.")

    if len(mu_with) == 0 or len(mu_without) == 0:
        print(f"❌ Failed for {attr_name}")
        return None

    attr_vector = mu_with.mean(dim=0) - mu_without.mean(dim=0)
    return attr_vector

# Compute vectors
attributes_to_find = ['Smiling', 'Male', 'Eyeglasses', 'Young']
attr_vectors = {}

print("Computing attribute vectors (this may take a minute)...\n")
for attr in attributes_to_find:
    vec = get_attribute_vector(model, attr_loader, attr)
    if vec is not None:
        attr_vectors[attr] = vec
        print(f"✅ {attr} SUCCESS\n")

print(f"Total successful attributes: {len(attr_vectors)}")

@torch.no_grad()
def attribute_grid(model, loader, attr_vectors, strengths=[-1.5, -0.75, 0, 0.75, 1.5]):
    if not attr_vectors:
        print("No attributes available. Cannot create grid.")
        return

    x, _ = next(iter(loader))
    base_face = x[0:1]   # Change 0 to try different faces

    n_attrs = len(attr_vectors)
    n_strengths = len(strengths)

    fig, axes = plt.subplots(n_attrs, n_strengths, figsize=(n_strengths*4, n_attrs*4))

    if n_attrs == 1:
        axes = np.array([axes])

    for row, (attr_name, vec) in enumerate(attr_vectors.items()):
        for col, strength in enumerate(strengths):
            modified = apply_attribute(model, base_face, vec, strength)
            ax = axes[row, col]
            ax.imshow(modified[0].permute(1, 2, 0).clip(0, 1))
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(attr_name, fontsize=14, rotation=0, labelpad=80, va='center')
            if row == 0:
                ax.set_title(f"{strength:+.1f}", fontsize=12)

    plt.suptitle("Attribute Manipulation Grid - Week 6 Assignment", fontsize=18)
    plt.tight_layout()
    plt.show()

# Generate the grid
attribute_grid(model, loader, attr_vectors)

# ================== PART 3: LATENT ARITHMETIC ==================

@torch.no_grad()
def show_arithmetic(model, test_face, attr_vectors):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(test_face[0].permute(1, 2, 0).clip(0,1))
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Single attribute: Smile
    if 'Smiling' in attr_vectors:
        smile_img = apply_attribute(model, test_face, attr_vectors['Smiling'], strength=1.2)
        axes[1].imshow(smile_img[0].permute(1, 2, 0).clip(0,1))
        axes[1].set_title("+ Smile")
        axes[1].axis('off')

    # Another single (e.g. Male or Eyeglasses)
    if 'Eyeglasses' in attr_vectors:
        glasses_img = apply_attribute(model, test_face, attr_vectors['Eyeglasses'], strength=1.5)
        axes[2].imshow(glasses_img[0].permute(1, 2, 0).clip(0,1))
        axes[2].set_title("+ Eyeglasses")
        axes[2].axis('off')

    # Combined (Smile + Young or Smile + Male)
    if 'Smiling' in attr_vectors and 'Young' in attr_vectors:
        combo_vec = attr_vectors['Smiling'] + attr_vectors['Young'] * 0.8
        combo_img = apply_attribute(model, test_face, combo_vec, strength=1.0)
        axes[3].imshow(combo_img[0].permute(1, 2, 0).clip(0,1))
        axes[3].set_title("+ Smile + Young")
        axes[3].axis('off')
    elif 'Smiling' in attr_vectors and 'Male' in attr_vectors:
        combo_vec = attr_vectors['Smiling'] + attr_vectors['Male'] * 0.7
        combo_img = apply_attribute(model, test_face, combo_vec, strength=1.0)
        axes[3].imshow(combo_img[0].permute(1, 2, 0).clip(0,1))
        axes[3].set_title("+ Smile + Male")
        axes[3].axis('off')

    plt.suptitle("Latent Space Arithmetic - Multiple Attributes", fontsize=16)
    plt.tight_layout()
    plt.show()

# Run it
x, _ = next(iter(loader))
test_face = x[5:6]   # Change number to try different faces
show_arithmetic(model, test_face, attr_vectors)
