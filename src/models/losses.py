import torch
import torch.nn as nn
import torch.nn.functional as F

def berhu_loss(pred: torch.Tensor, target: torch.Tensor,
               threshold_fraction: float = 0.2) -> torch.Tensor:
    """
    BerHu (reverse Huber) loss — implementation from scratch.
    Better than L1/L2 for depth: L1 for small errors, L2 for large.
    """
    # Only compute on valid pixels (target > 0)
    # For relative depth from MiDaS, target might be [0, 1]
    mask = target > 0
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
    pred_v = pred[mask]
    target_v = target[mask]
    
    diff = torch.abs(pred_v - target_v)
    with torch.no_grad():
        c = threshold_fraction * diff.max()
    
    if c == 0:
        return diff.mean()
        
    l1_mask = diff <= c
    l2_mask = ~l1_mask
    
    loss = torch.empty_like(diff)
    loss[l1_mask] = diff[l1_mask]
    loss[l2_mask] = (diff[l2_mask]**2 + c**2) / (2 * c + 1e-8)
    
    return loss.mean()

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Dice loss for segmentation.
    Works alongside CrossEntropyLoss to handle class imbalance.
    
    pred:   (B, C, H, W) logits
    target: (B, H, W)   long tensor
    """
    num_classes = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    
    # One-hot encode target
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (pred_soft * target_oh).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice_per_class.mean()

from torchvision import models

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss based on VGG-11 feature maps.
    Lighter than VGG-16 for faster training and smaller download.
    """
    def __init__(self, device="mps"):
        super(PerceptualLoss, self).__init__()
        print("Initializing Perceptual Loss (switching to VGG-11)...")
        vgg = models.vgg11(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        # VGG-11 Slices
        self.slice1 = vgg[:2]   # Relu1_1
        self.slice2 = vgg[2:5]  # Relu2_1
        self.slice3 = vgg[5:10] # Relu3_1
        self.slice4 = vgg[10:15]# Relu4_1
        
    def forward(self, x, y):
        h1_x = self.slice1(x)
        h1_y = self.slice1(y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        h4_x = self.slice4(h3_x)
        h4_y = self.slice4(h3_y)
        
        loss = F.l1_loss(h1_x, h1_y) + \
               F.l1_loss(h2_x, h2_y) + \
               F.l1_loss(h3_x, h3_y) + \
               F.l1_loss(h4_x, h4_y)
        return loss
