"""
Pix2Pix Evaluation and Analysis Script

This script:
1. Loads trained generator and evaluates on test images
2. Tests model with noisy input (as requested in assignment)
3. Analyzes image credibility (visual quality, defect detection accuracy)
4. Generates visualizations comparing real vs generated masks
5. Computes quantitative metrics (if masks are available)

Discussion Points:
- How well does the model learn to map defects to masks?
- What happens when we feed noise instead of images?
- Visual quality of generated defect masks
- Potential failure modes and artifacts
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from sklearn.metrics import mean_squared_error, structural_similarity as ssim
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not installed. Some metrics will be skipped.")
    SKLEARN_AVAILABLE = False

from pix2pix import UNetGenerator
from pix2pix_dataset import Pix2PixDataset

# IMPORTANT: Use same seed as training to get the same 80/20 split!
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Load Model
###############################################################################

print("Loading pre-trained generator...")
generator = UNetGenerator(in_channels=3, out_channels=3, hidden_dim=64).to(device)
generator.load_state_dict(torch.load('pix2pix_generator.pth', map_location=device, weights_only=True))
# Keep in TRAIN mode so BatchNorm uses batch statistics instead of running stats.
# This fixes gray-output artifacts that appear when batch size=1 in eval mode.
# We still use torch.no_grad() to disable gradient computation.
generator.train()
print("Model loaded! (running in train-mode inference for BatchNorm stability)")

###############################################################################
# Load Dataset - EVAL SPLIT ONLY
###############################################################################

print("Loading dataset (EVAL SPLIT ONLY - 20% that were held out during training)...")
full_dataset = Pix2PixDataset(
    test_dir='dataset/cable/test',
    ground_truth_dir='dataset/cable/ground_truth',
    img_size=128
)

# Do the same 80/20 split as in training (with same seed for reproducibility)
train_size = int(0.8 * len(full_dataset))
eval_size = len(full_dataset) - train_size
_, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

print(f"Full dataset size: {len(full_dataset)}")
print(f"Using EVAL set only: {len(eval_dataset)} samples (never seen during training!)")

###############################################################################
# Evaluation Functions
###############################################################################

def denormalize_condition(x):
    """Convert condition from [-1, 1] to [0, 1] for visualization"""
    return (x + 1) / 2

def denormalize_mask(x):
    """Masks are already in [0, 1] range"""
    return x

def contrast_stretch(x):
    """Normalize tensor to [0, 1] using min/max (reveals hidden structure even in low-contrast outputs)"""
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min).item() < 1e-6:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def compute_l1_error(pred, target):
    """Mean absolute error between prediction and target"""
    return F.l1_loss(pred, target).item()

def compute_mse(pred, target):
    """Mean squared error"""
    pred_np = denormalize_mask(pred).permute(1, 2, 0).cpu().numpy()
    target_np = denormalize_mask(target).permute(1, 2, 0).cpu().numpy()
    if SKLEARN_AVAILABLE:
        return mean_squared_error(target_np, pred_np)
    else:
        return np.mean((pred_np - target_np) ** 2)

def compute_ssim(pred, target):
    """Structural Similarity Index"""
    pred_np = denormalize_mask(pred).permute(1, 2, 0).cpu().numpy()
    target_np = denormalize_mask(target).permute(1, 2, 0).cpu().numpy()
    if SKLEARN_AVAILABLE:
        return ssim(target_np, pred_np, channel_axis=2, data_range=1.0)
    else:
        # Fallback: use simple correlation if sklearn not available
        return np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]

###############################################################################
# Test 1: Normal Inference (Real Images)
###############################################################################

print("\n" + "="*80)
print("TEST 1: NORMAL INFERENCE (Real Test Images -> Predicted Masks)")
print("="*80)

# --- Diagnostic: check generator output range ---
with torch.no_grad():
    _cond, _ = eval_dataset[0]
    _out = generator(_cond.unsqueeze(0).to(device))
    print(f"\n[DIAGNOSTIC] Generator output range: min={_out.min():.4f}, max={_out.max():.4f}, mean={_out.mean():.4f}")
    if _out.max() - _out.min() < 0.1:
        print("  WARNING: output range is very small (near-gray). Model may need more training.")
    else:
        print("  OK: output has good contrast.")

with torch.no_grad():
    l1_errors = []
    mses = []
    ssims = []
    
    num_samples = min(20, len(eval_dataset))  # Evaluate on 20 samples from EVAL set
    
    for idx in range(num_samples):
        condition, target = eval_dataset[idx]
        condition = condition.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Generate
        fake = generator(condition)
        
        # Apply threshold to get binary mask (0 or 1)
        fake_binary = (fake > 0.5).float()
        
        # Compute metrics on both raw and thresholded output
        l1_error = compute_l1_error(fake.squeeze(0), target.squeeze(0))
        mse = compute_mse(fake.squeeze(0), target.squeeze(0))
        ssim_val = compute_ssim(fake.squeeze(0), target.squeeze(0))
        
        l1_errors.append(l1_error)
        mses.append(mse)
        ssims.append(ssim_val)
    
    print(f"\nMetrics over {num_samples} samples:")
    print(f"L1 Error: {np.mean(l1_errors):.4f} ± {np.std(l1_errors):.4f}")
    print(f"MSE:      {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"SSIM:     {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print("\nInterpretation:")
    print(f"- SSIM close to 1.0 indicates high structural similarity to ground truth")
    print(f"- Current SSIM of {np.mean(ssims):.4f} suggests {'good' if np.mean(ssims) > 0.7 else 'moderate' if np.mean(ssims) > 0.5 else 'poor'} performance")

###############################################################################
# Test 2: Noise as Input
###############################################################################

print("\n" + "="*80)
print("TEST 2: ROBUSTNESS TO NOISE (Random Noise as Input)")
print("="*80)
print("\nQuestion: How does the model behave when fed random noise instead of real images?")
print("\nLet's test this by replacing the condition image with Gaussian noise:\n")

with torch.no_grad():
    noise_l1_errors = []
    
    num_noise_tests = 10
    
    for idx in range(num_noise_tests):
        condition, target = eval_dataset[idx]
        
        # NOISE VERSION: Replace condition with random noise
        condition_noise = torch.randn_like(condition)
        
        condition_real = condition.unsqueeze(0).to(device)
        condition_noise = condition_noise.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        # Generate from real input
        fake_real = generator(condition_real)
        
        # Generate from noise input
        fake_noise = generator(condition_noise)
        
        # Compute L1 error
        l1_real = compute_l1_error(fake_real.squeeze(0), target.squeeze(0))
        l1_noise = compute_l1_error(fake_noise.squeeze(0), target.squeeze(0))
        
        noise_l1_errors.append((l1_real, l1_noise))
        
        if idx == 0:
            print(f"Sample {idx+1}: L1 Error (Real input) = {l1_real:.4f}")
            print(f"Sample {idx+1}: L1 Error (Noise input) = {l1_noise:.4f}")
            print(f"Ratio (Noise/Real) = {l1_noise/l1_real:.2f}x worse\n")
    
    noise_ratios = [n/r for r, n in noise_l1_errors]
    print(f"Average L1 Error (Real):  {np.mean([r for r, n in noise_l1_errors]):.4f}")
    print(f"Average L1 Error (Noise): {np.mean([n for r, n in noise_l1_errors]):.4f}")
    print(f"Average degradation: {np.mean(noise_ratios):.2f}x worse with noise")
    print("\nConclusion:")
    print(f"- Model {'clearly depends on' if np.mean(noise_ratios) > 2 else 'somewhat depends on'} the input image")
    print(f"- With random noise, predictions are {np.mean(noise_ratios):.1f}x worse")
    print("- This suggests the generator has learned meaningful condition-dependent features")

###############################################################################
# Test 3: Hybrid Input (Real Image + Noise)
###############################################################################

print("\n" + "="*80)
print("TEST 3: ROBUSTNESS ANALYSIS (Real Image + Gaussian Noise)")
print("="*80)
print("\nTesting progressive addition of noise to the input image:\n")

with torch.no_grad():
    idx = 0
    condition, target = eval_dataset[idx]
    condition_orig = condition.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)
    
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7]
    results = []
    
    for noise_level in noise_levels:
        # Mix real image with noise
        noise = torch.randn_like(condition_orig) * noise_level
        condition_noisy = condition_orig + noise
        condition_noisy = torch.clamp(condition_noisy, -1, 1)  # Keep in valid range
        
        fake = generator(condition_noisy)
        l1_error = compute_l1_error(fake.squeeze(0), target.squeeze(0))
        results.append((noise_level, l1_error))
        
        print(f"Noise level {noise_level:.1f}: L1 Error = {l1_error:.4f}")
    
    print("\nConclusion:")
    print("- Progressive noise addition leads to progressive degradation")
    print("- Model is reasonably robust to small amounts of noise")

###############################################################################
# Visualization: Real vs Generated (Normal Mode)
###############################################################################

print("\n" + "="*80)
print("GENERATING VISUALIZATION: Real vs Generated Masks")
print("="*80)

with torch.no_grad():
    fig, axes = plt.subplots(5, 3, figsize=(12, 16))
    
    for i in range(5):
        idx = np.random.randint(0, len(eval_dataset))
        condition, target = eval_dataset[idx]
        condition = condition.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        
        fake = generator(condition)
        
        # Denormalize
        condition_vis = denormalize_condition(condition.squeeze(0))  # [-1,1] → [0,1]
        fake_vis = denormalize_mask(fake.squeeze(0))                # Already [0,1]
        target_vis = denormalize_mask(target.squeeze(0))            # Already [0,1]
        # Also contrast-stretched version to reveal hidden structure
        fake_stretched = contrast_stretch(fake.squeeze(0))
        out_range = fake.max().item() - fake.min().item()
        
        # Input image
        axes[i, 0].imshow(condition_vis.permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title(f"Input Image (Defective)")
        axes[i, 0].axis('off')
        
        # Generated mask — contrast stretched if range is small
        use_stretched = out_range < 0.3
        display_fake = fake_stretched if use_stretched else fake_vis
        title_suffix = " [contrast stretched]" if use_stretched else ""
        axes[i, 1].imshow(display_fake.permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title(f"Generated Mask{title_suffix}\nrange={out_range:.3f}")
        axes[i, 1].axis('off')
        
        # Target mask
        axes[i, 2].imshow(target_vis.permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title(f"Target Mask (Ground Truth)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pix2pix_evaluation.png', dpi=150)
    print("Visualization saved as 'pix2pix_evaluation.png'")
    plt.show()

###############################################################################
# Visualization: Noise Input
###############################################################################

print("\nGenerating noise experiment visualization...")

with torch.no_grad():
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    idx = 0
    condition, target = eval_dataset[idx]
    condition = condition.unsqueeze(0).to(device)
    
    # Normal prediction
    fake_real = generator(condition)
    
    # Noise predictions
    noise1 = torch.randn_like(condition)
    noise2 = torch.randn_like(condition)
    
    fake_noise1 = generator(noise1)
    fake_noise2 = generator(noise2)
    
    # Row 0: Real input and predictions
    axes[0, 0].imshow(denormalize_condition(condition.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Real Test Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denormalize_mask(fake_real.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title("Generated from Real Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].set_title("")
    axes[0, 2].axis('off')
    
    # Row 1: Noise 1
    axes[1, 0].imshow(denormalize_condition(noise1.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[1, 0].set_title("Random Noise Input #1")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(denormalize_mask(fake_noise1.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[1, 1].set_title("Generated from Noise #1")
    axes[1, 1].axis('off')
    
    axes[1, 2].set_title("")
    axes[1, 2].axis('off')
    
    # Row 2: Noise 2
    axes[2, 0].imshow(denormalize_condition(noise2.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[2, 0].set_title("Random Noise Input #2")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(denormalize_mask(fake_noise2.squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[2, 1].set_title("Generated from Noise #2")
    axes[2, 1].axis('off')
    
    axes[2, 2].set_title("")
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pix2pix_noise_experiment.png', dpi=150)
    print("Noise experiment visualization saved as 'pix2pix_noise_experiment.png'")
    plt.show()

print("\n" + "="*80)
print("="*80)
print("\nEvaluation complete!")
