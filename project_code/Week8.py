import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from torchvision import datasets, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
dataset = datasets.CelebA(root="./data", split="train", download=False, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
x0, _ = next(iter(loader))
x0 = x0.to(device)

class ForwardDiffusion:
    """Reusable forward diffusion pipeline (Milestone 3)"""
    def __init__(self, T=1000, schedule="linear", device="cpu"):
        self.T = T
        self.device = device
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, T, device=device)
        elif schedule == "cosine":
            def cosine_betas(T, s=0.008):
                t = torch.arange(T + 1, device=device)
                f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
                alphas_bar = f_t / f_t[0]
                return torch.clip(1 - alphas_bar[1:] / alphas_bar[:-1], max=0.999)
            betas = cosine_betas(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        alphas = 1 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.sqrt_ab = torch.sqrt(self.alphas_bar)
        self.sqrt_1m_ab = torch.sqrt(1 - self.alphas_bar)

    def q_sample(self, x0, t):
        """Closed-form: xt = √ᾱ_t x0 + √(1-ᾱ_t) ε"""
        eps = torch.randn_like(x0)
        sqrt_ab_t = self.sqrt_ab[t].view(-1, 1, 1, 1) if isinstance(t, (int, torch.Tensor)) and t.dim() == 0 else self.sqrt_ab[t]
        sqrt_1m_t = self.sqrt_1m_ab[t].view(-1, 1, 1, 1) if isinstance(t, (int, torch.Tensor)) and t.dim() == 0 else self.sqrt_1m_ab[t]
        return sqrt_ab_t * x0 + sqrt_1m_t * eps, eps

    def trajectory(self, x0, n_frames=100):
        """Full dissolution sequence for GIF"""
        ts = torch.linspace(0, self.T - 1, n_frames).long().to(self.device)
        frames = [x0]
        for t in ts[1:]:
            xt, _ = self.q_sample(x0, t.view(1))
            frames.append(xt)
        return ts, frames

#  VERIFICATION 
fd = ForwardDiffusion(T=1000, schedule="linear", device=device)

# t=0 check
xt0, _ = fd.q_sample(x0, torch.tensor([0], device=device))
print(f"t=0 error: {(x0 - xt0).abs().max():.2e}")  # should be ~0

# x_T check
xT, _ = fd.q_sample(x0, torch.tensor([fd.T - 1], device=device))
print(f"x_T mean: {xT.mean():.4f} (≈0)")
print(f"x_T std : {xT.std():.4f} (≈1)")

# FULL DISSOLUTION GIF 
ts, frames = fd.trajectory(x0, n_frames=100)
imgs = []
for xt in frames:
    arr = xt[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    imgs.append((arr * 255).astype("uint8"))
imageio.mimsave("diffusion_destruction.gif", imgs, fps=12)
print("Saved: diffusion_destruction.gif")

# Grid visualization
sample_steps = [0, 50, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(1, len(sample_steps), figsize=(18, 3))
for i, t in enumerate(sample_steps):
    xt, _ = fd.q_sample(x0, torch.tensor([t], device=device))
    axes[i].imshow(xt[0].cpu().permute(1, 2, 0).clamp(0, 1))
    axes[i].set_title(f"t={t}")
    axes[i].axis("off")
plt.suptitle("Controlled Destruction (Linear Schedule)")
plt.show()

#  Linear vs Cosine 
fd_cos = ForwardDiffusion(T=1000, schedule="cosine", device=device)
