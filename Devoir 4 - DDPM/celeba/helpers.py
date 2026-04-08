# source : @pankratozzi (kaggle)
# source : alexandre st-georges
# see great original article https://huggingface.co/blog/annotated-diffusion for details

# ── Imports ───────────────────────────────────────────────────────────────────
import os
from inspect import isfunction

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize, RandomHorizontalFlip

from PIL import Image


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ── Utility functions ─────────────────────────────────────────────────────────

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ── Beta schedules ────────────────────────────────────────────────────────────

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# ── Diffusion schedule precomputations ────────────────────────────────────────

timesteps = 50  # 300

betas = linear_beta_schedule(timesteps=timesteps)
# betas = cosine_beta_schedule(timesteps=timesteps)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# q(x_t | x_{t-1})
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ── Transforms ────────────────────────────────────────────────────────────────

transform = Compose([
    RandomHorizontalFlip(),
    Resize(64),
    CenterCrop(64),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1),
])

valid_transform = Compose([
    Resize(64),
    CenterCrop(64),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1),
])

reverse_transform = Compose([
    Lambda(lambda t: torch.clamp(t, -1.0, 1.0)),
    Lambda(lambda t: (t + 1) / 2),
    Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
    Lambda(lambda t: t * 255.),
    Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
    ToPILImage(),
])


# ── Forward diffusion ─────────────────────────────────────────────────────────

def q_sample(x_start, t, noise=None):
    """q(x_t | x_0) — add noise to image at timestep t (nice property)."""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# ── Training loss ─────────────────────────────────────────────────────────────

def p_losses(denoise_model, x_start, t, cond=None, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    if cond is not None:
        predicted_noise = denoise_model(x_noisy, t, cond)
    else:
        predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# ── Reverse diffusion (sampling) ──────────────────────────────────────────────

@torch.no_grad()
def p_sample(model, x, t, t_index, cond=None):
    """One denoising step: p(x_{t-1} | x_t). Equation 11."""
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    if cond is not None:
        pred = model(x, t, cond)
    else:
        pred = model(x, t)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape, cond=None):
    """Full denoising loop from pure noise to image. Algorithm 2."""
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, cond=cond)

    return [reverse_transform(image) for image in img]


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, cond=None):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), cond=cond)
