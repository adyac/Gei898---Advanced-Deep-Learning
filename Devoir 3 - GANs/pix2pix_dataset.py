"""
Dataset loader for pix2pix with paired images.

Expects structure:
    test/
        defect_type_1/
            image1.png
            image2.png
            ...
        defect_type_2/
            ...
    ground_truth/
        defect_type_1/
            image1_mask.png
            image2_mask.png
            ...
        ...

The loader pairs test images with corresponding ground_truth masks.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class Pix2PixDataset(Dataset):
    """
    Dataset for pix2pix training.
    
    Pairs input images (test/) with target images (ground_truth/).
    Both are resized to specified size.
    
    Args:
        test_dir (str): Path to test directory containing defects
        ground_truth_dir (str): Path to ground_truth directory containing masks
        img_size (int): Image size (default 128x128)
        defect_types (list): List of defect types to include (e.g., ['bent_lead', 'cut_lead', ...])
    """
    def __init__(self, test_dir, ground_truth_dir, img_size=128, defect_types=None):
        self.test_dir = Path(test_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.img_size = img_size
        self.image_pairs = []
        
        # Default defect types (cable dataset - exclude 'good' from training)
        if defect_types is None:
            defect_types = ['bent_wire', 'cable_swap', 'combined', 'cut_inner_insulation', 
                           'cut_outer_insulation', 'missing_cable', 'missing_wire', 'poke_insulation']
        
        # Find all paired images
        for defect_type in defect_types:
            test_defect_dir = self.test_dir / defect_type
            gt_defect_dir = self.ground_truth_dir / defect_type
            
            if not test_defect_dir.exists() or not gt_defect_dir.exists():
                print(f"Warning: {defect_type} not found in both directories")
                continue
            
            # Get all test images
            test_images = sorted([f for f in test_defect_dir.iterdir() if f.suffix in ['.png', '.jpg']])
            gt_images = sorted([f for f in gt_defect_dir.iterdir() if f.suffix in ['.png', '.jpg']])
            
            # Pair images (assume same filenames or ordering)
            for test_img, gt_img in zip(test_images, gt_images):
                self.image_pairs.append((str(test_img), str(gt_img)))
        
        print(f"Found {len(self.image_pairs)} paired images")
        
        # Transformations for CONDITION (input test images) - normalize to [-1, 1]
        self.transform_cond = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ])
        
        # Transformations for TARGET (ground truth masks) - keep in [0, 1] for sigmoid
        self.transform_target = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0, 1] - no normalization
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        test_path, gt_path = self.image_pairs[idx]
        
        # Load images
        test_img = Image.open(test_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')  # Ground truth masks as 3-channel
        
        # Apply transforms (separate for condition and target)
        test_tensor = self.transform_cond(test_img)  # [-1, 1]
        gt_tensor = self.transform_target(gt_img)     # [0, 1]
        
        return test_tensor, gt_tensor  # condition, target


def get_pix2pix_dataloader(test_dir, ground_truth_dir, batch_size=32, img_size=128, num_workers=0, shuffle=True):
    """
    Create a DataLoader for pix2pix training.
    
    Args:
        test_dir (str): Path to test directory
        ground_truth_dir (str): Path to ground_truth directory
        batch_size (int): Batch size
        img_size (int): Image size
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle
    
    Returns:
        DataLoader instance
    """
    dataset = Pix2PixDataset(test_dir, ground_truth_dir, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


###############################################################################
# Losses for Pix2Pix
###############################################################################

class Pix2PixLoss(nn.Module):
    """
    Combined loss for pix2pix:
        Loss = L_GAN + lambda * L_L1
    where:
        L_GAN: Adversarial loss (Hinge loss - more stable for GANs)
        L_reconstruction: Reconstruction loss (BCE for binary masks + L1 for smoothness)
    """
    def __init__(self, lambda_l1=5.0, use_hinge_loss=True):
        super(Pix2PixLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.use_hinge_loss = use_hinge_loss
        self.gan_loss = nn.BCEWithLogitsLoss() if not use_hinge_loss else None
        self.bce_loss = nn.BCELoss()  # For mask reconstruction (sigmoid output [0,1])
        self.l1_loss = nn.L1Loss()
    
    def generator_loss(self, fake_outputs, target_images, generated_images):
        """
        Generator loss: fool the discriminator + reconstruct target masks
        
        Args:
            fake_outputs: Discriminator output for generated images (should be high = real)
            target_images: Real target mask images [0, 1]
            generated_images: Generated mask images from generator [0, 1]
        
        Returns:
            Total generator loss, GAN loss, reconstruction loss
        """
        batch_size = fake_outputs.shape[0]
        
        # Adversarial loss: fool discriminator (make it think generated images are real)
        if self.use_hinge_loss:
            # Hinge loss: max(0, 1 - D(fake)) - more stable for GANs
            gan_loss = torch.mean(torch.relu(1.0 - fake_outputs))
        else:
            # BCE loss (original)
            real_labels = torch.ones_like(fake_outputs)
            gan_loss = self.gan_loss(fake_outputs, real_labels)
        
        # Reconstruction: BCE for binary mask classification + L1 for smoothness
        bce_loss = self.bce_loss(generated_images, target_images)
        l1_loss = self.l1_loss(generated_images, target_images)
        
        # Combined loss: GAN loss + weighted reconstruction (BCE is primary for masks, L1 for smoothness)
        total_loss = gan_loss + self.lambda_l1 * (bce_loss + 0.5 * l1_loss)
        
        return total_loss, gan_loss, bce_loss
    
    def discriminator_loss(self, real_outputs, fake_outputs, label_smooth=0.0):
        """
        Discriminator loss: classify real as real, fake as fake
        
        Args:
            real_outputs: Discriminator output for real images (should be high = real)
            fake_outputs: Discriminator output for fake images (should be low = fake)
            label_smooth: Label smoothing factor. real_labels = 1 - label_smooth
                         This prevents discriminator from being overconfident
        
        Returns:
            Total discriminator loss
        """
        batch_size = real_outputs.shape[0]
        
        if self.use_hinge_loss:
            # Hinge loss: max(0, 1 - D(real)) + max(0, 1 + D(fake))
            # Real images should have high D score (>1 is good)
            real_loss = torch.mean(torch.relu(1.0 - real_outputs))
            # Fake images should have low D score (<-1 is good)
            fake_loss = torch.mean(torch.relu(1.0 + fake_outputs))
        else:
            # BCE loss with label smoothing
            # Real labels are (1 - label_smooth) instead of 1
            real_label_val = 1.0 - label_smooth
            real_labels = torch.full_like(real_outputs, real_label_val)
            real_loss = self.gan_loss(real_outputs, real_labels)
            
            fake_labels = torch.zeros_like(fake_outputs)
            fake_loss = self.gan_loss(fake_outputs, fake_labels)
        
        # Average loss
        total_loss = (real_loss + fake_loss) / 2
        
        return total_loss
