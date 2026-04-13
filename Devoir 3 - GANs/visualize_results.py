"""
Visualization function to display transistor images with defects and their predicted vs actual masks.
Shows 5 examples in a single figure with 3 columns (input, generated, ground truth).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from pix2pix import UNetGenerator
from pix2pix_dataset import Pix2PixDataset


def visualize_predictions(num_samples=5, model_path='pix2pix_generator.pth', device='cuda'):
    """
    Visualize generator predictions: input images, generated masks, and ground truth masks.
    
    Args:
        num_samples (int): Number of examples to display (default 5)
        model_path (str): Path to trained generator weights
        device (str): Device to use ('cuda' or 'cpu')
    
    Returns:
        fig: matplotlib figure object
    """
    
    # Load dataset
    print("Loading dataset...")
    dataset = Pix2PixDataset(
        test_dir='dataset/cable/test',
        ground_truth_dir='dataset/cable/ground_truth',
        img_size=128
        # defect_types will use default (all cable defects)
    )
    
    # Load generator
    print(f"Loading model from {model_path}...")
    generator = UNetGenerator(in_channels=3, out_channels=3, hidden_dim=64).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("✓ Model loaded!")
    
    # Denormalize function (condition is [-1,1], targets/output are [0,1])
    def denormalize_input(x):
        """Condition images: convert from [-1, 1] to [0, 1]"""
        return (x + 1) / 2
    
    def denormalize_mask(x):
        """Target/generated masks: already in [0, 1]"""
        return x
    
    # Create figure with num_samples rows and 3 columns
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    # Handle case where num_samples=1 (axes is 1D)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    with torch.no_grad():
        for idx in range(num_samples):
            # Get random sample
            sample_idx = np.random.randint(0, len(dataset))
            condition, target = dataset[sample_idx]
            
            # Move to device
            condition_gpu = condition.unsqueeze(0).to(device)
            target_gpu = target.unsqueeze(0).to(device)
            
            # Generate prediction
            generated = generator(condition_gpu)
            
            # Denormalize for visualization
            condition_vis = denormalize_input(condition).cpu().numpy()
            generated_vis = denormalize_mask(generated.squeeze(0)).cpu().numpy()
            target_vis = denormalize_mask(target).cpu().numpy()
            
            # Column 0: Input transistor image (with defects)
            axes[idx, 0].imshow(condition_vis.transpose(1, 2, 0))
            axes[idx, 0].set_title(f'Sample {idx+1}: Input Transistor (Defective)', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Column 1: Generated mask (predicted by model)
            axes[idx, 1].imshow(generated_vis.transpose(1, 2, 0))
            axes[idx, 1].set_title(f'Sample {idx+1}: Generated Mask (Predicted)', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Column 2: Ground truth mask
            axes[idx, 2].imshow(target_vis.transpose(1, 2, 0))
            axes[idx, 2].set_title(f'Sample {idx+1}: Ground Truth Mask', fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
            
            print(f"  Sample {idx+1}/{num_samples} ✓")
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'pix2pix_predictions_{num_samples}_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    """
    Run this script to visualize 5 predictions:
    
    python visualize_results.py
    """
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Visualize 5 samples
    fig = visualize_predictions(num_samples=5, device=device)
