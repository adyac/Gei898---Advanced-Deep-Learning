"""
Pix2Pix training script.

Trains a U-Net Generator and PatchGAN Discriminator on paired images
from transistor dataset (test images -> ground truth masks).

To experiment with noise input, modified the code to support:
1. Normal mode: real test image as condition
2. Noise mode: random Gaussian noise as condition (comment/uncomment section below)
3. Hybrid mode: mix real image with noise
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pix2pix import UNetGenerator, PatchGANDiscriminator
from pix2pix_dataset import Pix2PixDataset, Pix2PixLoss

torch.manual_seed(0)

###############################################################################
# Hyperparameters
###############################################################################

n_epochs = 500
batch_size = 4  # Smaller batch for U-Net (memory intensive)
img_size = 128
lr = 0.0001
lr_disc = 0.000025  # Keep balanced (previous version)
beta_1 = 0.5
beta_2 = 0.999
lambda_l1 = 2.0  # Back to 2 (was working better)
label_smooth = 0.2  # Back to previous
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

###############################################################################
# Initialize Models
###############################################################################

generator = UNetGenerator(in_channels=3, out_channels=3, hidden_dim=64).to(device)
discriminator = PatchGANDiscriminator(in_channels=6, hidden_dim=64).to(device)

# Initialize weights
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)

generator.apply(init_weights)
discriminator.apply(init_weights)

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(beta_1, beta_2))

# Loss
pix2pix_loss = Pix2PixLoss(lambda_l1=lambda_l1, use_hinge_loss=True)  # Hinge loss is more stable

###############################################################################
# Load Data
###############################################################################

print("Loading dataset...")
dataset = Pix2PixDataset(
    test_dir='dataset/cable/test',
    ground_truth_dir='dataset/cable/ground_truth',
    img_size=img_size
)

# Split into train (80%) and eval (20%)
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

print(f"Dataset size: {len(dataset)}")
print(f"Train set: {len(train_dataset)} samples ({len(train_dataloader)} batches)")
print(f"Eval set: {len(eval_dataset)} samples ({len(eval_dataloader)} batches)")

###############################################################################
# Training Loop
###############################################################################

generator_losses = []
discriminator_losses = []
l1_losses = []
eval_gen_losses_curve = []  # Track eval loss per epoch
eval_l1_losses_curve = []   # Track eval loss per epoch

print(f"\nStarting training for {n_epochs} epochs...")
print(f"Batch size: {batch_size}, Image size: {img_size}x{img_size}")
print(f"Lambda L1: {lambda_l1}")

for epoch in range(n_epochs):
    gen_loss_epoch = 0
    disc_loss_epoch = 0
    l1_loss_epoch = 0
    disc_update_count = 0  # Track how many times we actually updated D
    step = 0
    
    for condition, target in tqdm(train_dataloader, leave=False, desc=f"Epoch {epoch+1}/{n_epochs}"):
        condition = condition.to(device)  # Real test image
        target = target.to(device)        # Real ground truth mask
        
        ###############################################################################
        # EXPERIMENT: Noise as condition (uncomment to test)
        ###############################################################################
        # To test how the system behaves with noise instead of real images:
        # Uncomment the line below to replace condition with random noise
        # condition = torch.randn_like(condition)
        
        ###############################################################################
        # Train Discriminator (with label smoothing to prevent it from converging too fast)
        ###############################################################################
        disc_optimizer.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            fake_images = generator(condition)
        
        # Discriminator output for real pairs (condition, target)
        real_output = discriminator(condition, target)
        
        # Discriminator output for fake pairs (condition, generated)
        fake_output = discriminator(condition, fake_images.detach())
        
        # Discriminator loss (with label smoothing: real labels are 0.8 instead of 1.0)
        # This prevents discriminator from being overconfident and helps GAN balance
        disc_loss = pix2pix_loss.discriminator_loss(real_output, fake_output, label_smooth=label_smooth)
        disc_loss.backward()
        disc_optimizer.step()
        
        disc_loss_epoch += disc_loss.item()
        disc_update_count += 1
        
        ###############################################################################
        # Train Generator
        ###############################################################################
        gen_optimizer.zero_grad()
        
        # Generate fake images
        fake_images = generator(condition)
        
        # Fool discriminator
        fake_output = discriminator(condition, fake_images)
        
        # Generator loss (adversarial + L1)
        total_gen_loss, gan_loss, l1_loss = pix2pix_loss.generator_loss(
            fake_output, target, fake_images
        )
        
        total_gen_loss.backward()
        gen_optimizer.step()
        
        gen_loss_epoch += total_gen_loss.item()
        l1_loss_epoch += l1_loss.item()
        step += 1
    
    # Average losses for epoch
    gen_loss_epoch /= len(train_dataloader)
    disc_loss_epoch /= disc_update_count if disc_update_count > 0 else 1
    l1_loss_epoch /= len(train_dataloader)
    
    generator_losses.append(gen_loss_epoch)
    discriminator_losses.append(disc_loss_epoch)
    l1_losses.append(l1_loss_epoch)
    
    # Compute eval loss for this epoch
    generator.eval()
    eval_gen_epoch = []
    eval_l1_epoch = []
    with torch.no_grad():
        for condition, target in eval_dataloader:
            condition = condition.to(device)
            target = target.to(device)
            fake = generator(condition)
            total_loss, gan_loss, l1_loss = pix2pix_loss.generator_loss(fake, target, fake)
            eval_gen_epoch.append(total_loss.item())
            eval_l1_epoch.append(l1_loss.item())
    
    eval_gen_losses_curve.append(np.mean(eval_gen_epoch))
    eval_l1_losses_curve.append(np.mean(eval_l1_epoch))
    generator.train()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: "
              f"GenLoss={gen_loss_epoch:.4f}, "
              f"EvalGenLoss={eval_gen_losses_curve[-1]:.4f}, "
              f"DiscLoss={disc_loss_epoch:.4f}, "
              f"L1Loss={l1_loss_epoch:.4f}")

print("Training complete!")

###############################################################################
# Compute Eval Losses on Full Dataset (at end, for reporting)
###############################################################################

print("\nFinal evaluation metrics:")
eval_gen_loss_avg = np.mean(eval_gen_losses_curve)
eval_l1_loss_avg = np.mean(eval_l1_losses_curve)
print(f"Eval Loss (Generator) - Average over all epochs: {eval_gen_loss_avg:.4f}")
print(f"Eval Loss (L1) - Average over all epochs: {eval_l1_loss_avg:.4f}")

###############################################################################
# Save Models
###############################################################################

torch.save(generator.state_dict(), 'pix2pix_generator.pth')
torch.save(discriminator.state_dict(), 'pix2pix_discriminator.pth')
print("Models saved!")

###############################################################################
# Visualization
###############################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Generator Loss (Train vs Eval curve)
axes[0].plot(generator_losses, label='Train', linewidth=2, alpha=0.8)
axes[0].plot(eval_gen_losses_curve, label='Eval', linewidth=2, alpha=0.8, linestyle='--')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Generator Loss (Train vs Eval)')
axes[0].legend()
axes[0].grid()

# L1 Reconstruction Loss (Train vs Eval curve)
axes[1].plot(l1_losses, label='Train', linewidth=2, alpha=0.8)
axes[1].plot(eval_l1_losses_curve, label='Eval', linewidth=2, alpha=0.8, linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('L1 Reconstruction Loss (Train vs Eval)')
axes[1].legend()
axes[1].grid()

# Discriminator Loss
axes[2].plot(discriminator_losses, label='Train', linewidth=2, alpha=0.8)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('Discriminator Loss')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.savefig('pix2pix_training_losses.png', dpi=150)
print("Loss plot saved as 'pix2pix_training_losses.png'")
plt.show()

###############################################################################
# Test on 10 random samples (better than 1 batch)
###############################################################################

print("\nGenerating samples from eval set (10 random samples)...")
generator.eval()

with torch.no_grad():
    # Generate 10 random samples
    fig, axes = plt.subplots(10, 3, figsize=(12, 30))
    
    for sample_idx in range(10):
        # Get random sample from eval dataset
        random_idx = np.random.randint(0, len(eval_dataset))
        condition, target = eval_dataset[random_idx]
        condition = condition.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Generate
        fake = generator(condition)
        
        # Denormalize for visualization
        def denormalize_condition(x):
            return (x + 1) / 2  # [-1,1] -> [0,1]
        
        def denormalize_mask(x):
            return x  # Already [0,1]
        
        # Condition (input)
        axes[sample_idx, 0].imshow(denormalize_condition(condition.squeeze(0)).permute(1, 2, 0).cpu().numpy())
        axes[sample_idx, 0].set_title(f"Sample {sample_idx+1}: Input")
        axes[sample_idx, 0].axis('off')
        
        # Generated
        axes[sample_idx, 1].imshow(denormalize_mask(fake.squeeze(0)).permute(1, 2, 0).cpu().numpy())
        out_range = fake.max().item() - fake.min().item()
        axes[sample_idx, 1].set_title(f"Generated (range={out_range:.3f})")
        axes[sample_idx, 1].axis('off')
        
        # Target
        axes[sample_idx, 2].imshow(denormalize_mask(target.squeeze(0)).permute(1, 2, 0).cpu().numpy())
        axes[sample_idx, 2].set_title(f"Ground Truth")
        axes[sample_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pix2pix_samples.png', dpi=150)
    print("Sample visualization saved as 'pix2pix_samples.png'")
    plt.show()

print("\nDone!")
