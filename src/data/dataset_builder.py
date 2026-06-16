import json
import cv2
import numpy as np
import os
import random
from pathlib import Path
from collections import Counter
import tqdm

# ── Folder → label mapping ────────────────────────────────────────────
STYLE_KEYWORDS = {
    'bohemian': 'bohemian',   'rustic': 'rustic',
    'scandinavian': 'scandinavian', 'industrial': 'industrial',
    'mid_century': 'mid_century',   'contemporary': 'contemporary',
    'minimalist': 'minimalist',     'farmhouse': 'farmhouse',
    'modern': 'modern',             'luxury': 'luxury',
    'transitional': 'transitional', 'eclectic': 'eclectic',
    'cozy': 'cozy',                 'apartment': 'modern',
}
ROOM_KEYWORDS = {
    'living_room': 'living_room', 'loft': 'living_room',
    'bedroom': 'bedroom',         'kitchen': 'kitchen',
    'bathroom': 'bathroom',       'dining_room': 'dining_room',
    'office': 'office',
}

def parse_folder_labels(folder_name: str) -> dict:
    name = folder_name.lower()
    style    = next((v for k,v in STYLE_KEYWORDS.items() if k in name), 'unknown')
    room     = next((v for k,v in ROOM_KEYWORDS.items()  if k in name), 'unknown')
    return {'style': style, 'room_type': room}

def extract_dominant_colors(img_path: Path, k: int = 5) -> list[str]:
    """
    KMeans on image pixels → top k dominant colors as hex strings.
    Implement KMeans from scratch (Lloyd's algorithm).
    """
    try:
        image = cv2.imread(str(img_path))
        if image is None: return []
        
        # Resize for speed
        image = cv2.resize(image, (150, 150))
        pixels = image.reshape(-1, 3).astype(float)
        
        # Initialize centers randomly
        indices = np.random.choice(len(pixels), k, replace=False)
        centers = pixels[indices]
        
        labels = np.zeros(len(pixels))
        
        for _ in range(20):  # 20 iterations is enough
            # dists shape: (N, k)
            dists = ((pixels[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = dists.argmin(axis=1)
            
            new_centers = np.array([pixels[labels == i].mean(axis=0)
                                    if (labels == i).any() else centers[i]
                                    for i in range(k)])
            
            if np.allclose(centers, new_centers, atol=1.0):
                break
            centers = new_centers
        
        # Sort by cluster size (most dominant first)
        counts = np.bincount(labels.astype(int), minlength=k)
        order = counts.argsort()[::-1]
        
        # Convert to hex (BGR → RGB hex)
        return [f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}"
                for c in centers[order]]
    except Exception as e:
        print(f"Error in extract_dominant_colors for {img_path}: {e}")
        return []

def build_room_metadata(raw_dir: Path, out_path: Path):
    """
    Scan all room folders → write rooms.jsonl
    """
    project_root = Path("/Users/snehapatel/visionary")
    entries = []
    
    # Folders to skip (IKEA folders)
    ikea_prefixes = ['ikea_']
    
    folders = [f for f in raw_dir.iterdir() if f.is_dir() and not any(f.name.startswith(p) for p in ikea_prefixes)]
    
    print(f"Scanning {len(folders)} folders in {raw_dir}...")
    
    for folder in folders:
        labels = parse_folder_labels(folder.name)
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.webp"))
        
        for img_path in tqdm.tqdm(images, desc=f"Processing {folder.name}"):
            rel_path = str(img_path.relative_to(project_root))
            
            # Dominant colors (Top 3)
            # Use k=5 but take top 3 as in previous implementation or as needed
            colors = extract_dominant_colors(img_path, k=5)[:3]
            
            entry = {
                "path": rel_path,
                "room_type": labels['room_type'],
                "style": labels['style'],
                "dominant_colors": colors
            }
            entries.append(entry)
            
    # Stratified Split (80/10/10) by room_type
    random.seed(42)
    room_entries = {}
    for entry in entries:
        rt = entry['room_type']
        if rt not in room_entries: room_entries[rt] = []
        room_entries[rt].append(entry)
        
    train_set, val_set, test_set = [], [], []
    for rt, rt_entries in room_entries.items():
        random.shuffle(rt_entries)
        n = len(rt_entries)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        
        for i, entry in enumerate(rt_entries):
            if i < n_train:
                entry['split'] = 'train'
                train_set.append(entry)
            elif i < n_train + n_val:
                entry['split'] = 'val'
                val_set.append(entry)
            else:
                entry['split'] = 'test'
                test_set.append(entry)
                
    # Save rooms.jsonl
    with open(out_path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
            
    # Print counts
    print("\n" + "─"*40)
    print(f"{'room_type':<20} | {'train':>6} | {'val':>6} | {'test':>6}")
    print("─"*40)
    for rt in sorted(room_entries.keys()):
        count_train = sum(1 for e in train_set if e['room_type'] == rt)
        count_val   = sum(1 for e in val_set   if e['room_type'] == rt)
        count_test  = sum(1 for e in test_set  if e['room_type'] == rt)
        print(f"{rt:<20} | {count_train:6} | {count_val:6} | {count_test:6}")
    print("─"*40)
    print(f"Total entries: {len(entries)}")

# ── JOB B: Furniture library ──────────────────────────────────────────
from src.geometry.backprojector import depth_to_pointcloud_np, clean_pointcloud_np, estimate_intrinsics, save_ply

def infer_furniture_style(image: np.ndarray) -> str:
    """
    Classical HSV-based style classification — no neural network.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_sat = hsv[:,:,1].mean()
    mean_val = hsv[:,:,2].mean()
    mean_hue = hsv[:,:,0].mean()
    
    if mean_sat < 40 and mean_val > 180:  return 'scandinavian'
    if mean_val < 100:                     return 'industrial'
    if mean_sat > 80 and 10 < mean_hue < 25: return 'bohemian'
    if mean_sat < 30:                      return 'minimalist'
    if 100 < mean_hue < 130:               return 'modern'
    return 'transitional'

def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Center at origin, scale longest dimension to 1.0.
    """
    if len(points) == 0: return points
    centroid = points.mean(axis=0)
    points -= centroid
    scale = np.abs(points).max()
    points /= (scale + 1e-8)
    return points

def build_furniture_library(
    raw_dir: Path,
    depth_model,           # trained FastDepthNet
    out_dir: Path,
    index_path: Path,
):
    """
    Job B: scan IKEA folders → build PLY library → build furniture_index.json
    """
    os.makedirs(out_dir, exist_ok=True)
    device = next(depth_model.parameters()).device
    
    IKEA_TYPE_MAP = {
        'ikea_beds': 'bed',       'ikea_chairs': 'chair',
        'ikea_tables': 'table',     'ikea_beds': 'bed',
        'ikea_lighting': 'lamp',    'ikea_rugs': 'rug',
        'ikea_storage': 'storage',  'ikea_curtains': 'curtain',
        'ikea_sofas': 'sofa',
    }
    
    index = []
    ikea_folders = [f for f in raw_dir.iterdir() if f.is_dir() and f.name.startswith('ikea_')]
    
    print(f"Processing {len(ikea_folders)} IKEA folders...")
    
    for folder in ikea_folders:
        ftype = IKEA_TYPE_MAP.get(folder.name, 'unknown')
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.webp"))
        
        type_dir = out_dir / ftype
        os.makedirs(type_dir, exist_ok=True)
        
        for img_path in tqdm.tqdm(images, desc=f"Processing {ftype}"):
            img_id = f"{ftype}_{img_path.stem}"
            image = cv2.imread(str(img_path))
            if image is None: continue
            
            # 1. Depth Inference
            depth = depth_model.infer(image, device=device)
            
            # 2. Back-project to PCD
            h, w = image.shape[:2]
            K = estimate_intrinsics(w, h)
            points, colors = depth_to_pointcloud_np(image, depth, K)
            
            # 3. Clean and Normalize
            points, colors = clean_pointcloud_np(points, colors)
            points = normalize_points(points)
            
            # 4. Classical Style & Color stats
            style = infer_furniture_style(image)
            
            # Color Histogram
            # (Re-use extract_dominant_colors logic)
            dom_colors = extract_dominant_colors(img_path, k=3)
            
            # 5. Save PCD
            ply_path = type_dir / f"{img_id}.ply"
            save_ply(str(ply_path), points, colors)
            
            # 6. Save Histogram (for style matching)
            # Re-implementing color histogram here for speed/convenience
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = np.zeros((16, 8, 8), dtype=np.float32)
            H = (hsv[:,:,0] / 180.0 * 16).astype(int).clip(0, 15)
            S = (hsv[:,:,1] / 256.0 *  8).astype(int).clip(0, 7)
            V = (hsv[:,:,2] / 256.0 *  8).astype(int).clip(0, 7)
            np.add.at(hist, (H.ravel(), S.ravel(), V.ravel()), 1)
            hist = hist / (hist.sum() + 1e-8)
            hist_path = type_dir / f"{img_id}_hist.npy"
            np.save(hist_path, hist.ravel())
            
            # 7. Add to index
            index.append({
                "id": img_id,
                "type": ftype,
                "style": style,
                "ply_path": str(ply_path.relative_to(Path("/Users/snehapatel/visionary"))),
                "thumbnail_path": str(img_path.relative_to(Path("/Users/snehapatel/visionary"))),
                "color_hist_path": str(hist_path.relative_to(Path("/Users/snehapatel/visionary"))),
                "dominant_colors": dom_colors,
                "bbox": {"min": list(points.min(axis=0)), "max": list(points.max(axis=0))},
                "product_url": f"https://www.ikea.com/us/en/search/?q={img_path.stem}",
                "product_name": img_path.stem.replace("_", " ").title(),
                "price": "$XXX"
            })
            
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
        
    print(f"Furniture library built with {len(index)} items. Saved to {index_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=["rooms", "furniture"], default="rooms")
    args = parser.parse_args()
    
    raw_dir = Path("/Users/snehapatel/visionary/data/datasets/raw")
    
    if args.job == "rooms":
        out_path = Path("/Users/snehapatel/visionary/data/annotations/rooms.jsonl")
        build_room_metadata(raw_dir, out_path)
    else:
        # Needs trained depth model
        from src.models.fast_depth_net import FastDepthNet
        import torch
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = FastDepthNet().to(device)
        model.load_state_dict(torch.load("/Users/snehapatel/visionary/models/depth_model.pth"))
        model.eval()
        
        out_dir = Path("/Users/snehapatel/visionary/data/furniture_library")
        index_path = Path("/Users/snehapatel/visionary/data/annotations/furniture_index.json")
        build_furniture_library(raw_dir, model, out_dir, index_path)
