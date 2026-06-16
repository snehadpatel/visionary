import numpy as np

def compute_depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    All metrics implemented from scratch with NumPy only.
    
    mask = gt > 0  (valid depth pixels)
    pred = pred[mask]
    gt   = gt[mask]
    
    threshold = np.maximum(pred/gt, gt/pred)
    delta1 = (threshold < 1.25   ).mean()
    delta2 = (threshold < 1.25**2).mean()
    delta3 = (threshold < 1.25**3).mean()
    
    abs_diff = np.abs(pred - gt)
    sq_diff  = (pred - gt) ** 2
    
    rmse    = np.sqrt(sq_diff.mean())
    abs_rel = (abs_diff / gt).mean()
    sq_rel  = (sq_diff  / gt).mean()
    
    # Avoid log(0)
    pred_log = np.log(np.maximum(pred, 1e-7))
    gt_log = np.log(np.maximum(gt, 1e-7))
    log_rms = np.sqrt(((pred_log - gt_log)**2).mean())
    
    return {
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'rmse': float(rmse),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'log_rms': float(log_rms)
    }

if __name__ == "__main__":
    # Test with dummy data
    pred = np.array([1.0, 2.0, 3.0])
    gt = np.array([1.1, 1.9, 3.2])
    metrics = compute_depth_metrics(pred, gt)
    print("Metrics dummy test:")
    for k, v in metrics.items():
        print(f"  {k:10}: {v:.4f}")
