import numpy as np

def compute_seg_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """
    Compute mIoU and Pixel Accuracy.
    pred, gt: (H, W) or (N, H, W) integer arrays
    """
    pred = pred.flatten()
    gt = gt.flatten()
    
    # Pixel Accuracy
    pixel_acc = (pred == gt).mean()
    
    # mIoU
    ious = []
    for c in range(num_classes):
        intersection = np.logical_and(pred == c, gt == c).sum()
        union = np.logical_or(pred == c, gt == c).sum()
        if union == 0:
            ious.append(float('nan')) # exclude if class not present in either
        else:
            ious.append(intersection / union)
            
    miou = np.nanmean(ious)
    
    return {
        'miou': float(miou),
        'pixel_accuracy': float(pixel_acc),
        'iou_per_class': [float(i) if not np.isnan(i) else None for i in ious]
    }

if __name__ == "__main__":
    # Dummy test
    pred = np.array([[0, 1], [1, 2]])
    gt = np.array([[0, 1], [2, 2]])
    metrics = compute_seg_metrics(pred, gt, 3)
    print("Segmentation Metrics Test:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  mIoU:           {metrics['miou']:.4f}")
    print(f"  IoU per class:  {metrics['iou_per_class']}")
