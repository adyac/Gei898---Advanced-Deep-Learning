# source : @pankratozzi (kaggle)
# source : alexandre st-georges
# see great original article https://huggingface.co/blog/annotated-diffusion for details
# same implementation in tf: https://keras.io/examples/generative/ddpm/

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

from model import UnetCas1
from helpers import (
    timesteps, transform, valid_transform, reverse_transform,
    p_losses, p_sample, sample,
)


# ── Hyperparameters ───────────────────────────────────────────────────────────

image_size = 64
channels = 3
batch_size = 32
epochs = 100
num_accumulation_steps = 16  # unused for now
dim_mults = (1, 2, 4)
loss_type = "huber"           # "huber" / "l1" / "l2"

lr = 1e-3
weight_decay = 1e-4
scheduler_patience = 2
scheduler_min_lr = 1e-6
scheduler_factor = 0.5

root = "./dataset/celeba_hq/"
use_pct_dataset = 100
workers = 4


# ── Device setup ──────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using '{device.upper()}' device")

os.makedirs("outputs", exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class CelebaDataset(Dataset):
    def __init__(self, data, transforms, dataset_percent=100):
        self.data = data[:int(len(data) * dataset_percent / 100)]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        image = Image.open(self.data[ix])
        return self.transforms(image)


# ── Evaluation metric (KID) ───────────────────────────────────────────────────

kid_transforms = Compose([
    reverse_transform,
    Resize(75),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class KidMetric(nn.Module):
    def __init__(self, gamma=3.0):
        super().__init__()
        self.backbone = torchvision.models.inception_v3(weights='DEFAULT', progress=True).to(device)
        self.backbone.dropout = nn.Identity()
        self.backbone.fc = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.gamma = gamma

    def polynomial_kernel(self, x1, x2):
        feature_dimensions = x1.size(1)
        return (x1 @ x2.transpose(0, 1) / feature_dimensions + 1.0) ** self.gamma

    def forward(self, real, generated):
        x1 = torch.cat([kid_transforms(t)[None] for t in real]).to(device)
        x2 = torch.cat([kid_transforms(t)[None] for t in generated]).to(device)

        with torch.no_grad():
            x1 = self.backbone(x1)
            x2 = self.backbone(x2)

        kernel_real = self.polynomial_kernel(x1, x1)
        kernel_generated = self.polynomial_kernel(x2, x2)
        kernel_cross = self.polynomial_kernel(x1, x2)

        n = x1.size(0)
        float_n = torch.tensor(n).float().to(device)
        mask = 1.0 - torch.eye(n).to(device)
        mean_kernel_real = torch.sum(kernel_real * mask) / (float_n * (float_n - 1.0))
        mean_kernel_generated = torch.sum(kernel_generated * mask) / (float_n * (float_n - 1.0))
        mean_kernel_cross = torch.mean(kernel_cross)
        return mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross


# ── Visualization ─────────────────────────────────────────────────────────────

def save_sample_grid(samples, epoch, output_dir="outputs"):
    n = len(samples)
    grid_size = math.ceil(math.sqrt(n))
    _, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axs = axs.flatten()
    for i, img in enumerate(samples):
        axs[i].imshow(img)
        axs[i].axis("off")
    for i in range(len(samples), len(axs)):
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"samples_epoch_{epoch:03d}.png"))
    plt.close()


def save_loss_curve(train_losses, valid_losses, output_dir="outputs"):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


# ── Model, optimizer, scheduler ───────────────────────────────────────────────

model = UnetCas1(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults,
    cond_dim=256,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
kidm = KidMetric().to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=scheduler_patience, min_lr=scheduler_min_lr, factor=scheduler_factor
)


# ── Data loaders ──────────────────────────────────────────────────────────────

image_paths = glob(root + "**/*.jpg", recursive=True)
train_paths, valid_paths = train_test_split(image_paths, test_size=0.1, shuffle=True)

train_ds = CelebaDataset(train_paths, transform, use_pct_dataset)
valid_ds = CelebaDataset(valid_paths, valid_transform, use_pct_dataset)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
print(f"Dataset size: {len(train_ds)} train / {len(valid_ds)} valid")


# ── Training loop ─────────────────────────────────────────────────────────────

train_losses, valid_losses, kid_metrics = [], [], []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")

    # train
    train_epoch_losses = []
    for step, batch in enumerate(train_dl, 1):
        model.train()
        optimizer.zero_grad()

        n = batch.size(0)
        batch = batch.to(device)
        t = torch.randint(0, timesteps, (n,), device=device).long()

        cond = model.encode(batch)                              # c' = F_enc(x_0)
        loss = p_losses(model, batch, t, cond=cond, loss_type=loss_type)
        loss.backward()
        optimizer.step()
        train_epoch_losses.append(loss.item())

        if step % 500 == 0:
            print(f"  step {step} | train loss: {loss.item():.5f}")

    # validate
    valid_epoch_losses, kid_epoch_metric = [], []
    for step, batch in enumerate(valid_dl, 1):
        model.eval()
        n = batch.size(0)
        batch = batch.to(device)
        t = torch.randint(0, timesteps, (n,), device=device).long()

        with torch.no_grad():
            cond = model.encode(batch)                          # c' = F_enc(x_0)
            loss = p_losses(model, batch, t, cond=cond, loss_type=loss_type)
        valid_epoch_losses.append(loss.item())

        if step % 50 == 0:
            img = torch.randn_like(batch).to(device)
            with torch.no_grad():
                for i in reversed(range(0, max(1, timesteps // 100))):
                    img = p_sample(model, img, torch.full((n,), i, device=device, dtype=torch.long), i, cond=cond)
            kid = kidm(batch, img)
            kid_epoch_metric.append(kid.item())
            print(f"  step {step} | valid loss: {loss.item():.5f} | kid: {kid.item():.5f}")

    train_mean = np.mean(train_epoch_losses)
    valid_mean = np.mean(valid_epoch_losses)
    print(f"Epoch {epoch+1} summary | train: {train_mean:.5f} | valid: {valid_mean:.5f} | kid: {np.mean(kid_epoch_metric):.5f}")

    train_losses.append(train_mean)
    valid_losses.append(valid_mean)
    if kid_epoch_metric:
        kid_metrics.append(np.mean(kid_epoch_metric))

    scheduler.step(valid_mean)
    # Use a validation batch as reference condition for generation
    ref = next(iter(valid_dl))[:batch_size].to(device)
    with torch.no_grad():
        sample_cond = model.encode(ref)
    save_sample_grid(sample(model, image_size=image_size, batch_size=batch_size, channels=channels, cond=sample_cond), epoch + 1)
    save_loss_curve(train_losses, valid_losses)


# ── Save & final inference ────────────────────────────────────────────────────

torch.save(model.state_dict(), "diffusion.pth")

ref = next(iter(valid_dl))[:batch_size].to(device)
with torch.no_grad():
    sample_cond = model.encode(ref)
samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels, cond=sample_cond)
save_sample_grid(samples, epochs)
