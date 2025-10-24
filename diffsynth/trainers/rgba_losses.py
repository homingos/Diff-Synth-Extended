"""
RGBA VAE Reconstruction Losses for Wan-Alpha Feature Merge Block Training.
Based on Wan-Alpha paper Equations 7-13.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class RGBAReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for RGBA VAE training (Wan-Alpha paper).

    Implements:
    - L_α: Alpha channel reconstruction loss (Equation 10)
    - L_rgb^s: Soft-rendered RGB reconstruction loss (Equation 11)
    - L_rgb^h: Hard-rendered RGB reconstruction loss (Equation 12)
    - L_vae: Total VAE loss (Equation 13)
    """

    def __init__(self, perceptual_weight=1.0, edge_weight=1.0):
        """
        Args:
            perceptual_weight: Weight for perceptual loss component
            edge_weight: Weight for edge loss component
        """
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight

        # VGG network for perceptual loss (Equation 8)
        # Use weights parameter instead of deprecated pretrained
        from torchvision.models import VGG16_Weights

        self.vgg = (
            torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            .features[:16]
            .eval()
        )
        for param in self.vgg.parameters():
            param.requires_grad = False

    def l1_loss(self, pred, target):
        """
        L1 reconstruction loss (Equation 7).
        L_rec = ||V̂ - V||
        """
        return F.l1_loss(pred, target)

    def perceptual_loss(self, pred, target):
        """
        Perceptual loss using VGG features (Equation 8).
        L_per = ||Φ(V̂) - Φ(V)||²
        """
        # Normalize from [-1, 1] to [0, 1] for VGG
        pred_norm = (pred + 1.0) / 2.0
        target_norm = (target + 1.0) / 2.0

        # Extract VGG features
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)

        # MSE between features
        return F.mse_loss(pred_features, target_features)

    def edge_loss(self, pred, target):
        """
        Edge gradient loss using Sobel operator (Equation 9).
        L_edge = ||S(V̂) - S(V)||
        """
        # Sobel filters
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype)
            .view(1, 1, 3, 3)
            .to(pred.device)
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype)
            .view(1, 1, 3, 3)
            .to(pred.device)
        )

        # Apply Sobel to each channel and average
        def sobel_gradient(img):
            # img shape: [B, C, H, W] or [B, C, T, H, W]
            if img.dim() == 5:
                # For video, process each frame
                B, C, T, H, W = img.shape
                img = img.reshape(B * C * T, 1, H, W)
            else:
                B, C, H, W = img.shape
                img = img.reshape(B * C, 1, H, W)

            grad_x = F.conv2d(img, sobel_x, padding=1)
            grad_y = F.conv2d(img, sobel_y, padding=1)
            gradient = torch.sqrt(grad_x**2 + grad_y**2)

            if img.dim() == 5:
                gradient = gradient.reshape(B, C, T, H, W)
            else:
                gradient = gradient.reshape(B, C, H, W)

            return gradient

        pred_edges = sobel_gradient(pred)
        target_edges = sobel_gradient(target)

        return F.l1_loss(pred_edges, target_edges)

    def composite_loss(self, pred, target):
        """
        Composite loss combining L1, perceptual, and edge losses.
        Used for L_α, L_rgb^s, and L_rgb^h (Equations 10, 11, 12).

        L = L_rec + L_per + L_edge
        """
        loss_rec = self.l1_loss(pred, target)
        loss_per = self.perceptual_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)

        total = (
            loss_rec + self.perceptual_weight * loss_per + self.edge_weight * loss_edge
        )
        return total, {
            "l1": loss_rec.item(),
            "perceptual": loss_per.item(),
            "edge": loss_edge.item(),
        }

    def forward(
        self,
        pred_alpha,
        gt_alpha,
        pred_rgb_soft,
        gt_rgb_soft,
        pred_rgb_hard,
        gt_rgb_hard,
    ):
        """
        Calculate total VAE loss (Equation 13).
        L_vae = L_α + L_rgb^s + L_rgb^h

        Args:
            pred_alpha: Predicted alpha video [B, C, T, H, W]
            gt_alpha: Ground truth alpha video [B, C, T, H, W]
            pred_rgb_soft: Predicted soft-rendered RGB [B, C, T, H, W]
            gt_rgb_soft: Ground truth soft-rendered RGB [B, C, T, H, W]
            pred_rgb_hard: Predicted hard-rendered RGB [B, C, T, H, W]
            gt_rgb_hard: Ground truth hard-rendered RGB [B, C, T, H, W]

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # L_α: Alpha channel loss (Equation 10)
        loss_alpha, alpha_components = self.composite_loss(pred_alpha, gt_alpha)

        # L_rgb^s: Soft-rendered RGB loss (Equation 11)
        loss_rgb_soft, soft_components = self.composite_loss(pred_rgb_soft, gt_rgb_soft)

        # L_rgb^h: Hard-rendered RGB loss (Equation 12)
        loss_rgb_hard, hard_components = self.composite_loss(pred_rgb_hard, gt_rgb_hard)

        # L_vae: Total loss (Equation 13)
        total_loss = loss_alpha + loss_rgb_soft + loss_rgb_hard

        # Detailed loss breakdown
        loss_dict = {
            "total": total_loss.item(),
            "alpha": loss_alpha.item(),
            "rgb_soft": loss_rgb_soft.item(),
            "rgb_hard": loss_rgb_hard.item(),
            "alpha_l1": alpha_components["l1"],
            "alpha_perceptual": alpha_components["perceptual"],
            "alpha_edge": alpha_components["edge"],
            "rgb_soft_l1": soft_components["l1"],
            "rgb_soft_perceptual": soft_components["perceptual"],
            "rgb_soft_edge": soft_components["edge"],
            "rgb_hard_l1": hard_components["l1"],
            "rgb_hard_perceptual": hard_components["perceptual"],
            "rgb_hard_edge": hard_components["edge"],
        }

        return total_loss, loss_dict
