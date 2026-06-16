import numpy as np
from typing import Optional, List
from scipy.spatial import KDTree
from dataclasses import dataclass
import cv2

UNVISITED = -2
NOISE = -1

def dbscan_3d(
    points: np.ndarray,   # (N, 3)
    eps: float = 0.08,    # 8cm neighborhood
    min_pts: int = 15,
) -> np.ndarray:           # labels (N,) — -1=noise, >=0 cluster id
    """
    DBSCAN clustering implementation from scratch.
    """
    N = len(points)
    if N == 0: return np.array([])
    
    tree = KDTree(points)
    labels = np.full(N, UNVISITED, dtype=int)
    
    def neighbors(i):
        return tree.query_ball_point(points[i], eps)
    
    cid = 0
    for i in range(N):
        if labels[i] != UNVISITED: continue
        
        nbrs = neighbors(i)
        if len(nbrs) < min_pts:
            labels[i] = NOISE
            continue
            
        labels[i] = cid
        seeds = set(nbrs) - {i}
        
        while seeds:
            q = seeds.pop()
            if labels[q] == NOISE: 
                labels[q] = cid
            if labels[q] != UNVISITED: 
                continue
                
            labels[q] = cid
            q_nbrs = neighbors(q)
            if len(q_nbrs) >= min_pts:
                seeds.update(q_nbrs)
        cid += 1
        
    return labels

@dataclass
class FurnitureCluster:
    label: int
    ftype: str              # sofa, chair, table, lamp, rug, storage, curtain
    centroid: np.ndarray       # (3,)
    bbox_min: np.ndarray       # (3,)
    bbox_max: np.ndarray       # (3,)
    volume: float
    height_above_floor: float
    points: np.ndarray
    colors: np.ndarray
    color_hist: np.ndarray       # (1024,) normalized HSV histogram
    mask: Optional[np.ndarray] = None

def extract_clusters(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    seg_map_flat: np.ndarray,    # (N,) class labels from UNet for these points
    floor_height: float,
) -> list[FurnitureCluster]:
    """
    Convert cluster labels into FurnitureCluster objects.
    """
    # Pascal VOC 21 Classes Mapping:
    # 0: background, 9: chair, 11: diningtable, 16: pottedplant, 18: sofa, 20: tvmonitor
    FURNITURE_CLASSES = {
        9: 'chair',
        11: 'table',
        18: 'sofa',
        16: 'lamp',    # Proxy for decor
        20: 'storage'  # Proxy for storage/TV units
    }
    
    clusters = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == NOISE: continue
        
        mask = (labels == label)
        pts = points[mask]
        cls = colors[mask]
        
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        dims = bbox_max - bbox_min
        volume = dims[0] * dims[1] * dims[2]
        centroid = pts.mean(axis=0)
        # Height heuristic: distance from floor
        # In Y-down coords, floor_height is larger than most points.
        # base_height = floor_height - bbox_max[1]
        height_above_floor = floor_height - bbox_max[1]
        
        print(f"DEBUG: Cluster {label}: vol={volume:.4f}, h_above={height_above_floor:.4f}")
        
        # Filtering heuristics
        # If it's too far below the "detected" floor, it might be the REAL floor that was missed.
        if volume < 0.0005: 
            print(f"DEBUG: Skipping cluster {label} - volume too small")
            continue
        
        # Classify type by majority vote
        votes = seg_map_flat[mask]
        majority_class = int(np.bincount(votes, minlength=21).argmax())
        
        print(f"DEBUG: Cluster {label}: majority_class={majority_class}")
        
        # FIX: Even if class is 0 (background), if it's a significant object above floor, 
        # we treat it as furniture to ensure it gets masked/redesigned.
        if majority_class not in FURNITURE_CLASSES:
            if volume > 0.05 and height_above_floor > 0.2:
                print(f"DEBUG: Including unlabeled cluster {label} as 'furniture' based on geometry.")
                ftype = "storage" # Default generic type for redesigning
            else:
                print(f"DEBUG: Skipping cluster {label} - class {majority_class} and not geometrically significant.")
                continue
        else:
            ftype = FURNITURE_CLASSES[majority_class]
        
        # Color Histogram (16x8x8 HSV)
        # Convert RGB [0,1] colors to HSV
        rgb_uint8 = (cls * 255).astype(np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        
        hist = np.zeros((16, 8, 8), dtype=np.float32)
        H = (hsv[:, :, 0] / 180 * 16).astype(int).clip(0, 15)
        S = (hsv[:, :, 1] / 256 * 8).astype(int).clip(0, 7)
        V = (hsv[:, :, 2] / 256 * 8).astype(int).clip(0, 7)
        np.add.at(hist, (H.ravel(), S.ravel(), V.ravel()), 1)
        hist = hist / (hist.sum() + 1e-8)
        
        clusters.append(FurnitureCluster(
            label=int(label),
            ftype=ftype,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            volume=float(volume),
            height_above_floor=float(height_above_floor),
            points=pts,
            colors=cls,
            color_hist=hist.ravel()
        ))
        
    return clusters
