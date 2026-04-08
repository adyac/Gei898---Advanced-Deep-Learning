# Conditional DDPM — pix2pix-style image translation via cUnet
# Conditioning: F_enc encodes a reference face image → FiLM into every ResNet block
# CFG: guidance_scale > 1 at inference; cfg_drop_prob randomly nulls condition during training

import importlib.util, pathlib
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob

# ── Import "Helpers 2.py" (space in name prevents normal import) ──────────────
_spec = importlib.util.spec_from_file_location(
    "helpers2",
    pathlib.Path(__file__).parent / "Helpers 2.py",
)
_h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_h)

timesteps       = _h.timesteps
transform       = _h.transform
valid_transform = _h.valid_transform
p_losses        = _h.p_losses   # p_losses(model, x, t, ..., condition=, cfg_drop_prob=)
p_sample        = _h.p_sample   # p_sample(model, x, t, t_idx, condition=, guidance_scale=)
sample          = _h.sample     # sample(model, image_size, ..., condition=, guidance_scale=)
# ─────────────────────────────────────────────────────────────────────────────

from model_cddpm import cUnet


# ── Hyperparameters ───────────────────────────────────────────────────────────
image_size = 64
channels   = 3
batch_size = 32
epochs     = 300
dim_mults  = (1, 2, 4)
cond_dim   = 256        # F_enc output dimension (condition vector c')
loss_type  = "huber"   # "huber" / "l1" / "l2"

use_cfg        = True   # classifier-free guidance
cfg_drop_prob  = 0.1    # fraction of steps where condition is nulled during training
guidance_scale = 2.5    # CFG scale at inference (1.0 = no guidance)
ema_decay      = 0.995  # 0 = disabled

lr                 = 2e-4
weight_decay       = 1e-4
scheduler_patience = 2
scheduler_min_lr   = 1e-6
scheduler_factor   = 0.5
kid_eval_interval  = 50

root     = "./data/celeba_hq/"
workers  = 4
do_train = True   # False = load checkpoint and run inference only
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using '{device.upper()}' device")

os.makedirs("outputs", exist_ok=True)


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters for smoother inference."""
    def __init__(self, model, decay=0.995):
        self.model  = model
        self.decay  = decay
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


# ── Visualisation helpers ─────────────────────────────────────────────────────

def save_loss_curve(train_losses, valid_losses, output_dir="outputs"):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "closs_curve.png"))
    plt.close()


def save_eval_grid(mdl, male_cond_paths, female_cond_paths, epoch,
                   n=10, output_dir="outputs", guidance_scale=1.0):
    """
    4-column grid:
      col 1 : male condition image
      col 2 : generated female (conditioned on col 1)
      col 3 : female condition image
      col 4 : generated male (conditioned on col 3)
    """
    mdl.eval()
    dev = next(mdl.parameters()).device

    # Raw condition images as tensors [n, C, H, W]
    male_cond_tensors   = torch.stack([
        valid_transform(Image.open(p).convert("RGB")) for p in male_cond_paths[:n]
    ]).to(dev)
    female_cond_tensors = torch.stack([
        valid_transform(Image.open(p).convert("RGB")) for p in female_cond_paths[:n]
    ]).to(dev)

    # cUnet.forward encodes condition via F_enc internally
    generated_females = sample(mdl, image_size=image_size, batch_size=n,
                               channels=channels, condition=male_cond_tensors,
                               guidance_scale=guidance_scale)
    generated_males   = sample(mdl, image_size=image_size, batch_size=n,
                               channels=channels, condition=female_cond_tensors,
                               guidance_scale=guidance_scale)

    male_cond_imgs   = [Image.open(p).convert("RGB").resize((image_size, image_size))
                        for p in male_cond_paths[:n]]
    female_cond_imgs = [Image.open(p).convert("RGB").resize((image_size, image_size))
                        for p in female_cond_paths[:n]]

    _, axs = plt.subplots(n, 4, figsize=(8, 2 * n))
    titles = ["Male (cond)", "Female (generated)", "Female (cond)", "Male (generated)"]
    for col, title in enumerate(titles):
        axs[0, col].set_title(title, fontsize=8)
    for row in range(n):
        for col, img in enumerate([male_cond_imgs[row], generated_females[row],
                                    female_cond_imgs[row], generated_males[row]]):
            axs[row, col].imshow(img)
            axs[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ceval_epoch_{epoch:03d}.png"))
    plt.close()


# ── Dataset ───────────────────────────────────────────────────────────────────

def split_paths(paths):
    """Split a path list 50/50 into (condition, target) subsets."""
    mid = len(paths) // 2
    return paths[:mid], paths[mid:]


class PairedCelebaDataset(Dataset):
    """
    Returns (target_img, condition_img) pairs for pix2pix-style training.
    Both directions are included:
      indices [0,   n) : target = female_target, condition = random male_cond
      indices [n, 2*n) : target = male_target,   condition = random female_cond
    """
    def __init__(self, male_target_paths, male_cond_paths,
                 female_target_paths, female_cond_paths, transforms):
        n = min(len(male_target_paths), len(male_cond_paths),
                len(female_target_paths), len(female_cond_paths))
        self.male_target_paths   = male_target_paths[:n]
        self.male_cond_paths     = male_cond_paths[:n]
        self.female_target_paths = female_target_paths[:n]
        self.female_cond_paths   = female_cond_paths[:n]
        self.n          = n
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


# ── Model ─────────────────────────────────────────────────────────────────────
#
# cUnet.forward(x, time, condition=None)
#   condition = raw image [B, C, H, W] → encoded by internal F_enc → [B, cond_dim]
#   condition = None                   → learned null_cond embedding (CFG uncond path)
#
# Every cResNetBlock in encoder, bottleneck, and decoder receives FiLM(c_t, c').

model = cUnet(
    dim                 = image_size,
    channels            = channels,
    dim_mults           = dim_mults,
    cond_dim            = cond_dim,
    resnet_block_groups = 4,
).to(device)

ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

model_path = os.path.join("outputs", "cdiffusion.pth")
ema_path   = os.path.join("outputs", "cema.pth")

if not do_train:
    assert os.path.isfile(model_path), \
        f"No saved model found at '{model_path}'. Train first (set do_train=True)."
    model.load_state_dict(torch.load(model_path, map_location=device))
    if ema is not None and os.path.isfile(ema_path):
        ema.shadow = torch.load(ema_path, map_location=device)
        ema.apply()
    model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=scheduler_patience, min_lr=scheduler_min_lr, factor=scheduler_factor
)


# ── Path discovery (always runs — val paths needed for final inference) ────────
male_train_paths   = glob(os.path.join(root, "train/male/**/*.jpg"),   recursive=True)
female_train_paths = glob(os.path.join(root, "train/female/**/*.jpg"), recursive=True)
male_val_paths     = glob(os.path.join(root, "val/male/**/*.jpg"),     recursive=True)
female_val_paths   = glob(os.path.join(root, "val/female/**/*.jpg"),   recursive=True)

male_train_cond,   male_train_target   = split_paths(male_train_paths)
female_train_cond, female_train_target = split_paths(female_train_paths)
male_val_cond,     male_val_target     = split_paths(male_val_paths)
female_val_cond,   female_val_target   = split_paths(female_val_paths)


# ── Training / validation loop ────────────────────────────────────────────────

if do_train:
    train_ds = PairedCelebaDataset(male_train_target, male_train_cond,
                                   female_train_target, female_train_cond, transform)
    valid_ds = PairedCelebaDataset(male_val_target, male_val_cond,
                                   female_val_target, female_val_cond, valid_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    print(f"Train: {len(train_ds)}  |  Val: {len(valid_ds)}")

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")

        # ── Train ─────────────────────────────────────────────────────────────
        train_epoch_losses = []
        for step, (target, condition) in enumerate(train_dl, 1):
            model.train()
            optimizer.zero_grad()

            target    = target.to(device)       # [B, C, H, W]  noisy target image
            condition = condition.to(device)    # [B, C, H, W]  raw condition image

            # Sample random diffusion timestep for each example
            t = torch.randint(0, timesteps, (target.size(0),), device=device).long()

            # p_losses adds noise, calls model(x_noisy, t, condition=condition)
            # cUnet.forward encodes condition via F_enc, then FiLMs every ResNet block.
            # cfg_drop_prob randomly replaces condition with None → null_cond embedding.
            drop = cfg_drop_prob if use_cfg else 0.0
            loss = p_losses(model, target, t,
                            loss_type=loss_type,
                            condition=condition,
                            cfg_drop_prob=drop)

            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()
            train_epoch_losses.append(loss.item())

            if step % 500 == 0:
                print(f"  Train Loss: {loss.item():.5f}")

        # ── Validation ────────────────────────────────────────────────────────
        valid_epoch_losses = []
        for step, (target, condition) in enumerate(valid_dl, 1):
            model.eval()
            target    = target.to(device)
            condition = condition.to(device)

            t = torch.randint(0, timesteps, (target.size(0),), device=device).long()
            with torch.no_grad():
                loss = p_losses(model, target, t,
                                loss_type=loss_type,
                                condition=condition)
            valid_epoch_losses.append(loss.item())

        # ── Epoch summary ──────────────────────────────────────────────────────
        train_mean = np.mean(train_epoch_losses)
        valid_mean = np.mean(valid_epoch_losses)
        print(f"Train epoch loss: {train_mean:.5f}  Valid epoch loss: {valid_mean:.5f}")

        train_losses.append(train_mean)
        valid_losses.append(valid_mean)
        scheduler.step(valid_mean)

        torch.save(model.state_dict(), model_path)
        if ema is not None:
            torch.save(ema.shadow, ema_path)
            ema.apply()

        gs = guidance_scale if use_cfg else 1.0
        save_eval_grid(model, male_val_cond, female_val_cond, epoch + 1, guidance_scale=gs)

        if ema is not None:
            ema.restore()
        save_loss_curve(train_losses, valid_losses)


# ── Final inference ───────────────────────────────────────────────────────────
if ema is not None:
    ema.apply()
gs = guidance_scale if use_cfg else 1.0
save_eval_grid(model, male_val_cond, female_val_cond, epochs, guidance_scale=gs)
