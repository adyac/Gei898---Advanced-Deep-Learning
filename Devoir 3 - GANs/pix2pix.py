"""
Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks
Paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-ToImage_Translation_With_CVPR_2017_paper.pdf

Architecture:
- Generator: U-Net with skip connections (256→128→64→32→16→32→64→128→256)
- Discriminator: PatchGAN - classifies 70x70 patches as real/fake

The generator condition on input image by concatenating it with latent features.
"""

import torch
from torch import nn

torch.manual_seed(0)


###############################################################################
# U-Net Generator (Conditional Generator)
###############################################################################

class UNetGenerator(nn.Module):
    """
    U-Net generator for pix2pix.
    
    Input: (B, 3, 128, 128) image
    Output: (B, 3, 128, 128) generated image
    
    The generator is conditioned on the input image via skip connections.
    """
    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64):
        super(UNetGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder blocks: downsampling
        self.enc1 = self._conv_block(in_channels, hidden_dim)           # 128→128
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = self._conv_block(hidden_dim, hidden_dim * 2)        # 64→64
        self.enc3 = self._conv_block(hidden_dim * 2, hidden_dim * 4)    # 32→32
        self.enc4 = self._conv_block(hidden_dim * 4, hidden_dim * 8)    # 16→16
        
        # Bottleneck
        self.bottleneck = self._conv_block(hidden_dim * 8, hidden_dim * 8)  # 8→8
        
        # Decoder blocks: upsampling with skip connections
        # Upsampling layer
        self.upconv4 = nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        # After concatenation with e4 (512), input is 256+512=768
        self.dec4 = self._conv_block(hidden_dim * 4 + hidden_dim * 8, hidden_dim * 4)   # 768 → 256
        
        self.upconv3 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        # After concatenation with e3 (256), input is 128+256=384
        self.dec3 = self._conv_block(hidden_dim * 2 + hidden_dim * 4, hidden_dim * 2)    # 384 → 128
        
        self.upconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        # After concatenation with e2 (128), input is 64+128=192
        self.dec2 = self._conv_block(hidden_dim + hidden_dim * 2, hidden_dim)        # 192 → 64
        
        self.upconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        # After concatenation with e1 (64), input is 64+64=128
        self.dec1 = self._conv_block(hidden_dim + hidden_dim, out_channels)      # 128 → 3
        
        self.sigmoid = nn.Sigmoid()  # Output to [0, 1] instead of Tanh [-1, 1]
    
    def _conv_block(self, in_c, out_c):
        """Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: (B, 3, 128, 128) input image
        Returns:
            out: (B, 3, 128, 128) generated image in [0, 1]
        """
        # Encoder (downsampling)
        e1 = self.enc1(x)                  # (B, 64, 128, 128)
        e2 = self.enc2(self.pool(e1))      # (B, 128, 64, 64)
        e3 = self.enc3(self.pool(e2))      # (B, 256, 32, 32)
        e4 = self.enc4(self.pool(e3))      # (B, 512, 16, 16)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4)) # (B, 512, 8, 8)
        
        # Decoder 4 (upsampling 8→16)
        d4 = self.upconv4(b)               # (B, 256, 16, 16)
        d4 = torch.cat([d4, e4], dim=1)    # Concatenate skip: (B, 512, 16, 16)
        d4 = self.dec4(d4)                 # (B, 256, 16, 16)
        
        # Decoder 3 (upsampling 16→32)
        d3 = self.upconv3(d4)              # (B, 128, 32, 32)
        d3 = torch.cat([d3, e3], dim=1)    # Concatenate skip: (B, 256, 32, 32)
        d3 = self.dec3(d3)                 # (B, 128, 32, 32)
        
        # Decoder 2 (upsampling 32→64)
        d2 = self.upconv2(d3)              # (B, 64, 64, 64)
        d2 = torch.cat([d2, e2], dim=1)    # Concatenate skip: (B, 128, 64, 64)
        d2 = self.dec2(d2)                 # (B, 64, 64, 64)
        
        # Decoder 1 (upsampling 64→128)
        d1 = self.upconv1(d2)              # (B, 64, 128, 128)
        d1 = torch.cat([d1, e1], dim=1)    # Concatenate skip: (B, 128, 128, 128)
        d1 = self.dec1(d1)                 # (B, 3, 128, 128)
        
        return self.sigmoid(d1)


###############################################################################
# PatchGAN Discriminator
###############################################################################

class PatchGANDiscriminator(nn.Module):
    """
    Discriminator for pix2pix.
    
    Instead of classifying entire images as real/fake, it classifies 70x70 patches,
    which encourages the generator to produce locally realistic details.
    
    Input: (B, 6, 128, 128) concatenated [real_image, generated_image]
    Output: (B, 1, 15, 15) patch-wise classification
    
    The output is a heatmap where each position corresponds to a 70x70 receptive field.
    """
    def __init__(self, in_channels=6, hidden_dim=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # Discriminator network: progressively downsamples to small patch scores
        self.disc = nn.Sequential(
            self._disc_block(in_channels, hidden_dim, use_bn=False, dropout_rate=0.0),      # 128 -> 64 (no dropout on first)
            self._disc_block(hidden_dim, hidden_dim * 2, dropout_rate=0.3),                 # 64 -> 32 (dropout 30%)
            self._disc_block(hidden_dim * 2, hidden_dim * 4, dropout_rate=0.3),             # 32 -> 16 (dropout 30%)
            self._disc_block(hidden_dim * 4, hidden_dim * 8, dropout_rate=0.3),             # 16 -> 8 (dropout 30%)
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=1),  # 8 -> 7 (final patch)
        )
    
    def _disc_block(self, in_channels, out_channels, use_bn=True, dropout_rate=0.0):
        """Discriminator block: Conv -> (BatchNorm) -> LeakyReLU -> (Dropout)"""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, y):
        """
        Forward pass.
        
        Args:
            x: (B, 3, 128, 128) real/condition image
            y: (B, 3, 128, 128) real or generated image
        Returns:
            out: (B, 1, patch_h, patch_w) patch-wise scores
        """
        # Concatenate condition and image
        xy = torch.cat([x, y], dim=1)  # (B, 6, 128, 128)
        return self.disc(xy)
