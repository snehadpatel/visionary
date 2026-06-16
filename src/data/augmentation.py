import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_depth_train_transforms(size=(256, 256)):
    return A.Compose([
        A.Resize(*size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # Normalization and ToTensorV2 are handled in common.
    ], additional_targets={'depth': 'mask'})

def get_seg_train_transforms(size=(384, 384)):
    return A.Compose([
        A.Resize(*size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    ], additional_targets={'mask': 'mask'})

def get_val_transforms(size=(256, 256)):
    return A.Compose([
        A.Resize(*size),
    ], additional_targets={'depth': 'mask', 'mask': 'mask'})
