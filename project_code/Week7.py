import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Load sample images (CelebA)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)  # download=True first time
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
x, _ = next(iter(loader))
x = x.to(device)

#  CONFIG
T = 1000
beta_start = 1e-4
beta_end = 0.02
schedule_type = "linear"   # or "sine" "cosine"

#  SCHEDULE
if schedule_type == "linear":
    betas = torch.linspace(beta_start, beta_end, T, device=device)
elif schedule_type == "cosine":
    def cosine_betas(T, s=0.008):
        t = torch.arange(T + 1, device=device)
        f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return torch.clip(betas, max=0.999)
    betas = cosine_betas(T)

alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)        
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

print(f"ᾱ_1 = {alphas_bar[0]:.6f}, ᾱ_{T} = {alphas_bar[-1]:.6f}")

#  VISUALIZE SCHEDULE 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(betas.cpu())
axes[0].set_title("β_t (noise per step)")
axes[1].plot(alphas_bar.cpu(), color="green")
axes[1].set_title("ᾱ_t (signal remaining)")
axes[2].plot((1 - alphas_bar).cpu(), color="orange")
axes[2].set_title("1 - ᾱ_t (noise fraction)")
plt.tight_layout()
plt.show()

#  CLOSED-FORM FORWARD 
def forward_diffusion(x0, t, sqrt_alphas_bar, sqrt_one_minus_alphas_bar):
    """One-shot x0 → xt"""
    eps = torch.randn_like(x0)
    sqrt_ab_t = sqrt_alphas_bar[t].view(-1, 1, 1, 1) if t.dim() == 0 else sqrt_alphas_bar[t]
    sqrt_1m_t = sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1) if t.dim() == 0 else sqrt_one_minus_alphas_bar[t]
    return sqrt_ab_t * x0 + sqrt_1m_t * eps, eps


t = torch.tensor([600], device=device)
x0_fixed = x[0:1]
samples = []
for _ in range(2000):
    eps = torch.randn_like(x0_fixed)
    samples.append(sqrt_alphas_bar[t] * x0_fixed + sqrt_one_minus_alphas_bar[t] * eps)
samples = torch.stack(samples)
print("Mean error:", (samples.mean(0) - sqrt_alphas_bar[t] * x0_fixed).abs().mean().item())
print("Var error:", (samples.var(0) - (1 - alphas_bar[t])).abs().mean().item())
