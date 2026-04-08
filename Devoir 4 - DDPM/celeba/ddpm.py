# source : @pankratozzi (kaggle)
# source : alexandre st-georges
# see great original article https://huggingface.co/blog/annotated-diffusion for details
# same implementation in tf: https://keras.io/examples/generative/ddpm/

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob

from model import Unet, KidMetric, ConditionMode, SkipMode, EncoderAttnMode
from helpers import (
    timesteps, transform, valid_transform,
    p_losses, p_sample, sample
)


class EMA:
    """Exponential Moving Average of model parameters for smoother inference."""
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].sub_((1 - self.decay) * (self.shadow[name] - param.data))

    def apply(self):
        self.backup = {name: param.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])


# --- Hyperparameters ---
image_size = 64
channels = 3
batch_size = 32
epochs = 300
num_accumulation_steps = 16  # unused for now
dim_mults = (1, 2, 4)
loss_type = "huber"           # "huber" / "l1" / "l2"
cond_mode = ConditionMode.ENCODER   # ConditionMode.OFF / ConditionMode.ENCODER
skip_mode = SkipMode.NONE          # SkipMode.NONE / SkipMode.LINEAR / SkipMode.CROSS
encoder_attn_mode = EncoderAttnMode.CROSS  # EncoderAttnMode.SELF / EncoderAttnMode.CROSS
use_cfg   = True                    # True = classifier-free guidance, False = always conditioned (current approach)
train     = True                    # set to False to load outputs/diffusion.pth and run inference only

lr = 2e-4
weight_decay = 1e-4
scheduler_patience = 2
scheduler_min_lr = 1e-6
scheduler_factor = 0.5
kid_eval_interval = 50
cfg_drop_prob  = 0.1        # fraction of training batches with condition dropped (used when use_cfg=True)
guidance_scale = 2.5        # CFG guidance strength at inference (used when use_cfg=True)
ema_decay      = 0.995      # EMA decay rate (0 = disabled)

root = "./data/celeba_hq/"
workers = 4
# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using '{device.upper()}' device")

os.makedirs("outputs", exist_ok=True)


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


def save_eval_grid(model, male_cond_paths, female_cond_paths, epoch, n=10, output_dir="outputs", guidance_scale=1.0):
    """
    Generates and saves a 4-column evaluation grid:
      col 1 : original male conditions
      col 2 : model output (female) given col 1 as condition
      col 3 : original female conditions
      col 4 : model output (male) given col 3 as condition
    """
    model.eval()
    dev = next(model.parameters()).device

    # Load n condition images as tensors for the model
    male_cond_tensors = torch.stack([
        valid_transform(Image.open(p).convert("RGB")) for p in male_cond_paths[:n]
    ]).to(dev)
    female_cond_tensors = torch.stack([
        valid_transform(Image.open(p).convert("RGB")) for p in female_cond_paths[:n]
    ]).to(dev)

    # Generate: pure noise → model → output image
    generated_females = sample(model, image_size=image_size, batch_size=n,
                                channels=channels, condition=male_cond_tensors, guidance_scale=guidance_scale)
    generated_males   = sample(model, image_size=image_size, batch_size=n,
                                channels=channels, condition=female_cond_tensors, guidance_scale=guidance_scale)

    # Load condition images as PIL for display (original file, resized)
    male_cond_imgs   = [Image.open(p).convert("RGB").resize((image_size, image_size))
                        for p in male_cond_paths[:n]]
    female_cond_imgs = [Image.open(p).convert("RGB").resize((image_size, image_size))
                        for p in female_cond_paths[:n]]

    _, axs = plt.subplots(n, 4, figsize=(8, 2 * n))
    titles = ["Male (cond)", " Female (generated)", "Female (cond)", "Male (generated)"]
    for col, title in enumerate(titles):
        axs[0, col].set_title(title, fontsize=8)
    for row in range(n):
        for col, img in enumerate([
            male_cond_imgs[row],
            generated_females[row],
            female_cond_imgs[row],
            generated_males[row],
        ]):
            axs[row, col].imshow(img)
            axs[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"eval_epoch_{epoch:03d}.png"))
    plt.close()


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults,
    cond_mode=cond_mode,
    cond_channels=channels if cond_mode != ConditionMode.OFF else None,
    skip_mode=skip_mode,
    encoder_attn_mode=encoder_attn_mode,
).to(device)

ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

model_path = os.path.join("outputs", "diffusion.pth")
ema_path = os.path.join("outputs", "ema.pth")
if not train:
    assert os.path.isfile(model_path), f"No saved model found at '{model_path}'. Train first (set train=True)."
    model.load_state_dict(torch.load(model_path, map_location=device))
    if ema is not None and os.path.isfile(ema_path):
        ema.shadow = torch.load(ema_path, map_location=device)
        ema.apply()
    model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
kidm = KidMetric().to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, min_lr=scheduler_min_lr, factor=scheduler_factor)

class CelebaDataset(Dataset):
    """Single-image dataset for unconditional training."""
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self, ix):
        return self.transforms(Image.open(self.data[ix]).convert("RGB"))


def split_paths(paths):
    """Split a path list 50/50 into (condition, target) subsets."""
    mid = len(paths) // 2
    return paths[:mid], paths[mid:]


class PairedCelebaDataset(Dataset):
    """
    Returns (target, condition) pairs for pix2pix-style training.
    Condition and target subsets are disjoint (each 50% of the gender pool).
    Combines both translation directions:
      indices [0,   n) : target = female_target, condition = random male_cond   (male -> female)
      indices [n, 2*n) : target = male_target,   condition = random female_cond (female -> male)
    """
    def __init__(self, male_target_paths, male_cond_paths,
                 female_target_paths, female_cond_paths, transforms):
        n = min(len(male_target_paths), len(male_cond_paths),
                len(female_target_paths), len(female_cond_paths))
        self.male_target_paths   = male_target_paths[:n]
        self.male_cond_paths     = male_cond_paths[:n]
        self.female_target_paths = female_target_paths[:n]
        self.female_cond_paths   = female_cond_paths[:n]
        self.n = n
        self.transforms = transforms

    def __len__(self):
        return self.n * 2

    def __getitem__(self, ix):
        if ix < self.n:
            target_path    = self.female_target_paths[ix]
            condition_path = self.male_cond_paths[random.randint(0, self.n - 1)]
        else:
            target_path    = self.male_target_paths[ix - self.n]
            condition_path = self.female_cond_paths[random.randint(0, self.n - 1)]
        return (
            self.transforms(Image.open(target_path).convert("RGB")),
            self.transforms(Image.open(condition_path).convert("RGB")),
        )


# path discovery always runs — val paths are needed for inference regardless of mode
if cond_mode == ConditionMode.OFF:
    all_train_paths = glob(os.path.join(root, "train/**/*.jpg"), recursive=True)
    all_val_paths   = glob(os.path.join(root, "val/**/*.jpg"),   recursive=True)
else:
    male_train_paths   = glob(os.path.join(root, "train/male/**/*.jpg"),   recursive=True)
    female_train_paths = glob(os.path.join(root, "train/female/**/*.jpg"), recursive=True)
    male_val_paths     = glob(os.path.join(root, "val/male/**/*.jpg"),     recursive=True)
    female_val_paths   = glob(os.path.join(root, "val/female/**/*.jpg"),   recursive=True)

    male_train_cond,   male_train_target   = split_paths(male_train_paths)
    female_train_cond, female_train_target = split_paths(female_train_paths)
    male_val_cond,     male_val_target     = split_paths(male_val_paths)
    female_val_cond,   female_val_target   = split_paths(female_val_paths)

if train:
    if cond_mode == ConditionMode.OFF:
        train_ds = CelebaDataset(all_train_paths, transform)
        valid_ds = CelebaDataset(all_val_paths,   valid_transform)
    else:
        train_ds = PairedCelebaDataset(male_train_target, male_train_cond,
                                        female_train_target, female_train_cond, transform)
        valid_ds = PairedCelebaDataset(male_val_target, male_val_cond,
                                        female_val_target, female_val_cond, valid_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    print(f"Train: {len(train_ds)}  |  Val: {len(valid_ds)}")

    train_losses, valid_losses, kid_metrics = [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")
        train_epoch_losses = []
        for step, batch in enumerate(train_dl, 1):
            model.train()
            optimizer.zero_grad()

            if cond_mode == ConditionMode.OFF:
                target    = batch.to(device)
                condition = None
            else:
                target, condition = batch
                target    = target.to(device)
                condition = condition.to(device)
            n = target.size(0)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (n,), device=device).long()

            drop = cfg_drop_prob if use_cfg else 0.0
            loss = p_losses(model, target, t, loss_type=loss_type, condition=condition, cfg_drop_prob=drop)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()
            train_epoch_losses.append(loss.item())

            if step % 500 == 0:
                print(f"Train Loss: {loss.item():.5f}")

        valid_epoch_losses, kid_epoch_metric = [], []
        condition = None
        num_valid_steps = len(valid_dl)
        for step, batch in enumerate(valid_dl, 1):
            model.eval()
            if cond_mode == ConditionMode.OFF:
                target    = batch.to(device)
                condition = None
            else:
                target, condition = batch
                target    = target.to(device)
                condition = condition.to(device)
            n = target.size(0)

            t = torch.randint(0, timesteps, (n,), device=device).long()
            with torch.no_grad():
                loss = p_losses(model, target, t, loss_type=loss_type, condition=condition)

            valid_epoch_losses.append(loss.item())

            should_eval_kid = (
                n >= 2 and num_valid_steps > 0 and
                (step % kid_eval_interval == 0 or step == num_valid_steps)
            )
            if should_eval_kid:
                if ema is not None:
                    ema.apply()
                gs = guidance_scale if use_cfg else 1.0
                img = torch.randn_like(target).to(device)
                with torch.no_grad():
                    for i in reversed(range(0, timesteps//1)):
                        img = p_sample(model, img, torch.full((n,), i, device=device, dtype=torch.long), i, condition=condition, guidance_scale=gs)
                kid = kidm(target, img)
                if ema is not None:
                    ema.restore()
                if kid is not None:
                    kid_epoch_metric.append(kid.item())
                    print(f"Valid Loss: {loss.item():.5f}, kid: {kid.item():.5f}")
                else:
                    print(f"Valid Loss: {loss.item():.5f}, kid: skipped")

        train_mean = np.mean(train_epoch_losses)
        valid_mean = np.mean(valid_epoch_losses)
        kid_mean = np.mean(kid_epoch_metric) if kid_epoch_metric else None
        kid_summary = f"{kid_mean:.5f}" if kid_mean is not None else "n/a"
        print(f"Train epoch loss: {train_mean:.5f}",
              f"Valid epoch loss: {valid_mean:.5f}",
              f"Kid epoch metric: {kid_summary}")
        train_losses.append(train_mean)
        valid_losses.append(valid_mean)
        scheduler.step(valid_mean)
        torch.save(model.state_dict(), os.path.join("outputs", "diffusion.pth"))
        if ema is not None:
            torch.save(ema.shadow, ema_path)
            ema.apply()
        gs = guidance_scale if use_cfg else 1.0
        if cond_mode == ConditionMode.OFF:
            samples = sample(model, image_size=image_size, batch_size=36, channels=channels)
            save_sample_grid(samples, epoch + 1)
        else:
            save_eval_grid(model, male_val_cond, female_val_cond, epoch + 1, guidance_scale=gs)
        if ema is not None:
            ema.restore()
        save_loss_curve(train_losses, valid_losses)

if ema is not None:
    ema.apply()
gs = guidance_scale if use_cfg else 1.0
if cond_mode == ConditionMode.OFF:
    samples = sample(model, image_size=image_size, batch_size=36, channels=channels)
    save_sample_grid(samples, epochs)
else:
    save_eval_grid(model, male_val_cond, female_val_cond, epochs, guidance_scale=gs)
