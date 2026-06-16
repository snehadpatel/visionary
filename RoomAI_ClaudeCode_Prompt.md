# RoomAI — Full End-to-End Claude Code Prompt
# Real-Time CV Room Redesign System
# Solo Dev | Mac M4 24GB | 30 Days | End-Sem CV Project

---

## MISSION

Build RoomAI: a real-time room redesign system where the user points a webcam
at their room, says what changes they want, and sees the redesigned version
overlaid live — under 1 second latency. Computer Vision is the star of this
product. Every architectural decision must serve one goal:

> **"User speaks → room transforms → feels like magic"**

This is NOT a photo upload + wait 20 seconds tool.
This is NOT another Stable Diffusion wrapper.
This IS a real-time CV pipeline where depth estimation, point cloud
reconstruction, segmentation, and AR overlay run in parallel threads to
create a seamless live experience.

---

## WHAT IS ALREADY BUILT (DO NOT REBUILD)

### ✅ Dataset (3,826 images, fully processed)
- IKEA + Unsplash + House scene images
- Deduplicated (~2,100 duplicates removed), CLIP quality filtered
- BLIP-2 auto-labeling complete:
  → caption, room_type, style, objects[], dominant_colors[]
- Location: `data/raw/`, `data/annotations/metadata.jsonl`

### ✅ 7 CV Inference Modules (MPS optimized, built and tested)
- SAM ViT-H (32×32 point grid, area>1%, stability>0.8, IoU>0.75)
- BLIP-2 (ViT-G/14 + Q-Former + OPT-2.7B) — annotation only, NOT inference
- MiDaS DPT-Large (ViT-L/16, 384×384)
- Stable Diffusion 1.5 + ControlNet depth v1.1
- Real-ESRGAN 2× (RRDBNet)
- CLIP (segmentation classification)
- LoRA scaffold (untrained)

### ✅ Backend Foundation
- FastAPI app with routes
- SQLAlchemy models
- Celery + Redis worker setup

### ✅ Frontend Foundation
- React Studio page
- API hooks
- Basic UI shell

### ❌ NOT BUILT — BUILD EVERYTHING BELOW IN ORDER

---

## PROJECT STRUCTURE (FULL)

```
roomai/
├── data/
│   ├── raw/{ikea/, unsplash/, house/}
│   ├── annotations/metadata.jsonl
│   ├── splits/{train.jsonl, val.jsonl, test.jsonl}
│   ├── furniture_library/{scandinavian/, industrial/, modern/, bohemian/, minimalist/}
│   │   └── {type}/{id}.ply + {id}_meta.json
│   └── furniture_index.json
│
├── src/
│   ├── depth/
│   │   ├── fast_depth_net.py        # Lightweight MobileNet-style depth model
│   │   ├── train_depth.py           # Train on NYU Depth v2
│   │   └── depth_evaluator.py       # delta1/2/3, RMSE, AbsRel, SqRel
│   │
│   ├── pointcloud/
│   │   ├── backprojector.py         # depth → 3D (FROM SCRATCH)
│   │   ├── cleaner.py               # voxel downsample + outlier removal
│   │   └── normal_estimator.py      # PCA-based normals (FROM SCRATCH)
│   │
│   ├── geometry/
│   │   ├── ransac_plane.py          # RANSAC plane fitting (FROM SCRATCH)
│   │   ├── dbscan_custom.py         # DBSCAN clustering (FROM SCRATCH)
│   │   └── icp.py                   # ICP alignment (FROM SCRATCH)
│   │
│   ├── segmentation/
│   │   ├── unet.py                  # UNet architecture (FROM SCRATCH)
│   │   ├── train_seg.py             # Train on ADE20K indoor subset
│   │   └── seg_evaluator.py         # mIoU, pixel accuracy
│   │
│   ├── scene/
│   │   ├── room_analyzer.py         # Full scene understanding orchestrator
│   │   ├── furniture_detector.py    # Cluster → furniture type classification
│   │   ├── floor_detector.py        # Floor/wall detection using RANSAC
│   │   └── room_graph.py            # Multi-room graph
│   │
│   ├── redesign/
│   │   ├── style_engine.py          # Classical feature matching (NO neural retrieval)
│   │   ├── furniture_placer.py      # Scale/rotate/translate + ICP refinement
│   │   ├── color_transfer.py        # Reinhard color transfer (FROM SCRATCH)
│   │   └── blend_renderer.py        # 3D → 2D projection + alpha blend
│   │
│   ├── realtime/
│   │   ├── capture_thread.py        # Thread 1: webcam capture 30fps
│   │   ├── process_thread.py        # Thread 2: adaptive processing
│   │   ├── render_thread.py         # Thread 3: display 30fps
│   │   └── temporal_smoother.py     # Kalman filter (FROM SCRATCH)
│   │
│   └── ui/
│       ├── opencv_window.py         # Full OpenCV display with controls
│       ├── voice_commands.py        # SpeechRecognition trigger handler
│       └── room_selector.py         # Multi-room navigation UI
│
├── ml/
│   └── training/
│       ├── finetune_sd_lora.py      # LoRA fine-tune on our dataset
│       └── train_controlnet.py      # ControlNet depth conditioning
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/{redesign.py, rooms.py, pointcloud.py, realtime.py}
│   │   ├── models/{room.py, job.py, user.py}
│   │   └── services/{pipeline_service.py, storage_service.py}
│   └── tests/
│
├── frontend/
│   └── src/
│       ├── pages/{Studio.tsx, RoomGraph.tsx, Upload.tsx}
│       └── components/
│           ├── RealtimeViewer.tsx
│           ├── StylePanel.tsx
│           ├── DepthViewer.tsx
│           ├── PointCloudViewer.tsx
│           ├── BeforeAfterSlider.tsx
│           ├── FloorPlan.tsx
│           ├── VoiceIndicator.tsx
│           └── EvaluationDashboard.tsx
│
├── config/
│   └── camera_intrinsics.npy
├── models/
│   ├── depth_model.pth
│   └── seg_model.pth
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_depth_training.ipynb
│   ├── 03_segmentation_training.ipynb
│   ├── 04_pointcloud_pipeline.ipynb
│   ├── 05_redesign_pipeline.ipynb
│   └── 06_realtime_demo.ipynb
└── requirements.txt
```

---

## MAC M4 GLOBAL CONFIG (apply everywhere)

```python
# config/m4_config.py
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Inference sizes (tuned for <1 sec total latency budget)
DEPTH_INFERENCE_SIZE   = (256, 256)    # ~40ms on M4
SEG_INFERENCE_SIZE     = (512, 512)    # ~150ms on M4
INPAINT_SIZE           = (512, 512)    # ~3-4s on M4 (async)

# Point cloud limits
MAX_POINTS_REALTIME    = 50_000        # 30fps target
MAX_POINTS_STATIC      = 500_000       # single image mode

# Training
TRAINING_DTYPE         = torch.float32 # MPS fp16 training is unstable
INFERENCE_DTYPE        = torch.float16 # fp16 safe for MPS inference

# Cache clearing
MPS_CACHE_CLEAR_EVERY  = 100           # steps during training
```

---

## BUILD ORDER — STRICT SEQUENCE (30 days)

---

### PHASE 1: DATASET FINALIZATION + DEPTH MODEL (Days 1-2)

#### `data/processors/dataset_finalizer.py`

```python
def finalize_and_split_dataset(metadata_path: str) -> DatasetStats:
    """
    1. Load all entries from data/annotations/metadata.jsonl
    2. Validate every entry — required fields:
       - caption (str, non-empty)
       - room_type (str) — if missing, parse from caption using keyword match
         keywords: {living_room: ['living', 'lounge', 'couch'],
                    bedroom: ['bed', 'sleep', 'pillow'],
                    kitchen: ['kitchen', 'cook', 'counter'],
                    bathroom: ['bath', 'shower', 'toilet'],
                    dining: ['dining', 'table', 'chairs']}
       - style (str) — if missing, infer from caption:
         keywords: {scandinavian: ['scandi', 'nordic', 'birch', 'light wood'],
                    industrial: ['metal', 'concrete', 'exposed', 'dark'],
                    minimalist: ['minimal', 'clean', 'simple', 'white'],
                    bohemian: ['boho', 'warm', 'earthy', 'terracotta'],
                    modern: ['modern', 'contemporary', 'sleek']}
       - dominant_colors (list) — if missing, recompute:
         * Load image → resize to 150×150
         * KMeans(n_clusters=5) on reshaped pixels
         * Return top 3 cluster centers as hex strings
       - objects (list) — if missing, set to []
    
    3. Build distribution report:
       {room_type: {living_room: N, bedroom: N, ...},
        style: {modern: N, scandinavian: N, ...},
        total: N, valid: N, fixed: N, dropped: N}
    
    4. Flag any style or room_type with < 50 samples (warn, don't drop)
    
    5. Stratified split by room_type (80/10/10):
       → data/splits/train.jsonl
       → data/splits/val.jsonl
       → data/splits/test.jsonl
    
    6. Print dataset card to stdout
    Returns: DatasetStats dataclass
    """

def build_furniture_pcd_library():
    """
    Extract 3D point clouds for every IKEA furniture image.
    
    For each IKEA image in data/raw/ikea/:
    1. Run MiDaS DPT-Large → depth map (384×384)
    2. Back-project to point cloud using estimate_intrinsics_from_fov()
    3. Clean: voxel_size=0.02, statistical outlier removal (nb=20, std=2.0)
    4. Normalize: center at origin, scale longest dimension to 1.0
    5. Save as: data/furniture_library/{style}/{type}/{id}.ply
    
    Build furniture_index.json:
    [
      {
        "id": "ikea_sofa_001",
        "type": "sofa",
        "style": "scandinavian",
        "colors": ["#E8DCC8", "#F5F0E8"],
        "ply_path": "data/furniture_library/scandinavian/sofa/ikea_sofa_001.ply",
        "thumbnail_path": "data/raw/ikea/sofa_001.jpg",
        "volume_m3": 1.2,
        "bbox": {"x": 2.1, "y": 0.8, "z": 0.9}
      }
    ]
    """
```

#### `src/depth/fast_depth_net.py`

```python
class FastDepthNet(nn.Module):
    """
    Lightweight encoder-decoder depth estimation network.
    Target: ~40ms inference on M4 at 256×256.
    
    Architecture:
    
    ENCODER (MobileNetV2-inspired, pretrained ImageNet weights):
    - Conv2d(3, 32, 3, stride=2) + BN + ReLU6     → 128×128
    - InvertedResidual blocks: [16, 24, 32, 64, 96, 160, 320]
      with expansion factors: [1,  6,  6,  6,  6,   6,   6]
    - Use torchvision.models.mobilenet_v2(pretrained=True).features
    - Extract features at 4 scales: /2, /4, /8, /16
    
    DECODER (skip connections):
    - UpConv(320, 160) + skip from /8  → 32×32
    - UpConv(160, 96)  + skip from /4  → 64×64
    - UpConv(96, 32)   + skip from /2  → 128×128
    - UpConv(32, 16)                    → 256×256
    - Conv2d(16, 1) + Sigmoid           → depth map [0,1]
    
    UpConv block: bilinear upsample × 2 → Conv2d → BN → ELU
    (NOT ConvTranspose2d — causes checkerboard artifacts)
    
    def berhu_loss(pred, target, threshold_fraction=0.2):
        '''
        BerHu loss — implement from scratch. No external loss functions.
        
        diff = |pred - target|
        c = threshold_fraction * diff.max()
        
        L1 region (diff <= c):  loss = diff
        L2 region (diff >  c):  loss = (diff² + c²) / (2c)
        
        Return mean over all valid pixels (target > 0)
        '''
    
    Training config:
    - Dataset: NYU Depth v2 (795 train / 654 test)
      Download: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
      Use HDF5 loader: each sample = (RGB 640×480, depth 640×480)
    - Resize to 256×256 during training
    - Augmentation: random horizontal flip, color jitter (brightness=0.2, contrast=0.2)
    - Optimizer: Adam(lr=1e-4)
    - Scheduler: CosineAnnealingLR(T_max=30)
    - Epochs: 30 (≈4 hours on M4)
    - Save best checkpoint by delta1 score
    - Checkpoint: models/depth_model.pth
    """
```

#### `src/depth/depth_evaluator.py`

```python
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
    log_rms = np.sqrt(((np.log(pred) - np.log(gt))**2).mean())
    
    return dict with all 7 metrics
    """
```

---

### PHASE 2: POINT CLOUD PIPELINE (Days 3-5) — CV CORE

#### `src/pointcloud/backprojector.py`

```python
def estimate_intrinsics_from_fov(W: int, H: int, fov_deg: float = 65.0) -> np.ndarray:
    """
    Estimate 3×3 camera intrinsics K from field of view.
    Implement from scratch (3 lines of math).
    
    fx = W / (2 * tan(radians(fov_deg) / 2))
    fy = fx * H / W
    cx = W / 2.0
    cy = H / 2.0
    
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """

def depth_to_pointcloud(
    rgb: np.ndarray,      # (H, W, 3) uint8
    depth: np.ndarray,    # (H, W) float32, metric meters
    K: np.ndarray,        # (3, 3) camera intrinsics
    max_depth: float = 8.0,
) -> o3d.geometry.PointCloud:
    """
    Back-projection — implement with NumPy only. No open3d backproject.
    
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Vectorized back-projection
    Z = depth
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]
    
    # Filter valid points
    valid = (Z > 0.1) & (Z < max_depth)
    
    # Stack to (N, 3)
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    colors = rgb[valid].astype(np.float64) / 255.0
    
    # Return colored open3d PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def scale_midas_depth_to_metric(
    relative_depth: np.ndarray,
    scale_factor: float = 3.5,
    shift_factor: float = 0.5,
) -> np.ndarray:
    """
    MiDaS outputs relative inverse depth. Convert to approximate metric.
    depth_metric = scale_factor / (relative_depth + shift_factor)
    Clamp to [0.1, 10.0] meters (indoor range)
    These defaults are tuned for typical indoor scenes.
    """
```

#### `src/pointcloud/cleaner.py`

```python
def clean_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    """
    Standard indoor point cloud cleaning pipeline.
    
    Step 1: Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    Step 2: Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    Step 3: Normal estimation (PCA-based — implement from scratch below)
    estimate_normals_pca(pcd, radius=0.1, max_nn=30)
    
    Step 4: Orient normals toward camera origin
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))
    
    Return cleaned pcd
    """
```

#### `src/pointcloud/normal_estimator.py`

```python
def estimate_normals_pca(
    pcd: o3d.geometry.PointCloud,
    radius: float = 0.1,
    max_nn: int = 30,
) -> np.ndarray:
    """
    PCA-based normal estimation — implement from scratch.
    
    points = np.asarray(pcd.points)
    tree = KDTree(points)
    normals = []
    
    For each point p_i:
        neighbors_idx = tree.query_ball_point(p_i, radius)
        cap at max_nn nearest
        
        if len(neighbors) < 3:
            normals.append([0, 1, 0])  # fallback: up direction
            continue
        
        # PCA on neighbor covariance
        neighbors_pts = points[neighbors_idx]
        centered = neighbors_pts - neighbors_pts.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Normal = eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]
        normals.append(normal)
    
    pcd.normals = o3d.utility.Vector3dVector(np.array(normals))
    return np.array(normals)
    """
```

---

### PHASE 3: GEOMETRY — RANSAC + DBSCAN (Days 4-5) — FROM SCRATCH

#### `src/geometry/ransac_plane.py`

```python
def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.015,
    n_iterations: int = 1000,
    min_inliers: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC plane fitting — pure NumPy. Do NOT use open3d.segment_plane().
    
    Algorithm:
    
    best_inliers = 0
    best_plane = None
    best_mask = None
    N = len(points)
    
    for _ in range(n_iterations):
        # 1. Sample 3 random non-collinear points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # 2. Compute plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-10:
            continue  # collinear points, skip
        normal = normal / norm_mag
        d = -np.dot(normal, p1)
        
        # 3. Point-to-plane distances (vectorized)
        distances = np.abs(points @ normal + d)
        inlier_mask = distances < threshold
        n_inliers = inlier_mask.sum()
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_plane = np.array([*normal, d])
            best_mask = inlier_mask
    
    if best_inliers < min_inliers:
        return None, np.zeros(N, dtype=bool)
    
    # Refit using all inliers via SVD (more accurate than 3-point fit)
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # smallest singular value → plane normal
    if normal[1] < 0:
        normal = -normal  # ensure normal points up for floor
    d = -normal @ centroid
    
    return np.array([*normal, d]), best_mask


def detect_room_geometry(pcd: o3d.geometry.PointCloud) -> dict:
    """
    Iterative plane detection to find floor and walls.
    
    points = np.asarray(pcd.points)
    remaining = points.copy()
    remaining_idx = np.arange(len(points))
    
    Iteration 1 — Find FLOOR:
        plane, mask = fit_plane_ransac(remaining)
        Check floor: |normal[1]| > 0.85  (mostly vertical normal = horizontal floor)
        floor_plane = plane
        floor_height = mean Y of inliers
        Remove floor inliers from remaining
    
    Iterations 2-4 — Find WALLS:
        plane, mask = fit_plane_ransac(remaining)
        Check wall: |normal[1]| < 0.3  (mostly horizontal normal = vertical wall)
        Collect up to 3 wall planes
        Remove wall inliers from remaining
    
    Furniture = remaining points after floor+walls removed
    
    Returns:
    {
        'floor_plane':    np.ndarray(4,),      # [a,b,c,d]
        'floor_height':   float,
        'wall_planes':    list of np.ndarray,
        'floor_pcd':      o3d.PointCloud,
        'structure_pcd':  o3d.PointCloud,      # floor + walls
        'furniture_pcd':  o3d.PointCloud,      # everything else
        'floor_normal':   np.ndarray(3,),
    }
    """
```

#### `src/geometry/dbscan_custom.py`

```python
UNVISITED = -2
NOISE     = -1

def dbscan_3d(
    points: np.ndarray,  # (N, 3)
    eps: float = 0.08,
    min_pts: int = 15,
) -> np.ndarray:          # labels (N,) — -1=noise, >=0 cluster id
    """
    DBSCAN — implement the expansion logic from scratch.
    Use scipy.spatial.KDTree for neighbor queries (allowed).
    Do NOT use sklearn.cluster.DBSCAN.
    
    tree = KDTree(points)
    N = len(points)
    labels = np.full(N, UNVISITED, dtype=int)
    
    def get_neighbors(i):
        return tree.query_ball_point(points[i], eps)
    
    cluster_id = 0
    for i in range(N):
        if labels[i] != UNVISITED:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_pts:
            labels[i] = NOISE
            continue
        
        labels[i] = cluster_id
        seed_set = set(neighbors) - {i}
        
        while seed_set:
            q = seed_set.pop()
            if labels[q] == NOISE:
                labels[q] = cluster_id
            if labels[q] != UNVISITED:
                continue
            labels[q] = cluster_id
            q_neighbors = get_neighbors(q)
            if len(q_neighbors) >= min_pts:
                seed_set.update(q_neighbors)
        
        cluster_id += 1
    
    return labels


@dataclass
class FurnitureCluster:
    label:      int
    type:       str            # sofa, chair, table, lamp, plant, cabinet, unknown
    centroid:   np.ndarray     # (3,) center in 3D
    bbox_min:   np.ndarray     # (3,)
    bbox_max:   np.ndarray     # (3,)
    volume_m3:  float
    height_m:   float          # height above floor
    pcd:        o3d.geometry.PointCloud
    color_hist: np.ndarray     # HSV histogram for style matching


def classify_clusters(
    labels: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    floor_height: float,
) -> list[FurnitureCluster]:
    """
    Convert cluster labels → FurnitureCluster objects.
    
    For each unique label >= 0:
    
    1. Extract cluster points and colors
    
    2. Bounding box:
       bbox_min = pts.min(axis=0)
       bbox_max = pts.max(axis=0)
       dims = bbox_max - bbox_min  # [width, height, depth]
    
    3. Metrics:
       volume  = dims[0] * dims[1] * dims[2]
       height  = bbox_min[1] - floor_height  # distance above floor
       
    4. Filter out non-furniture:
       height < 0.02 or height > 2.5 → skip (floor or ceiling)
       volume < 0.003 → skip (too small, noise)
       volume > 20.0  → skip (structural element)
    
    5. Classify by geometry:
       w, h, d = dims
       if h < 0.55 and max(w,d) > 0.7:
           type = 'table'
       elif h > 0.5 and min(w,d) > 0.4:
           if max(w,d) > 1.5:
               type = 'sofa'
           else:
               type = 'chair' or 'cabinet'
       elif h > 0.8 and min(w,d) < 0.35:
           type = 'lamp' or 'plant'
       else:
           type = 'unknown'
    
    6. Color histogram (for style matching):
       img_pixels = (colors * 255).astype(np.uint8)
       hsv = cv2.cvtColor(img_pixels.reshape(-1,1,3), cv2.COLOR_RGB2HSV)
       hist = cv2.calcHist([hsv], [0,1,2], None, [16,8,8], [0,180,0,256,0,256])
       hist = hist / hist.sum()  # normalize
    
    Return list of FurnitureCluster
    """
```

#### `src/geometry/icp.py`

```python
def icp_align(
    source: np.ndarray,     # (N, 3) points to transform
    target: np.ndarray,     # (M, 3) reference points
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Iterative Closest Point — implement from scratch. No open3d.registration.
    
    Uses scipy.spatial.KDTree for nearest neighbor queries.
    
    R_total = np.eye(3)
    t_total = np.zeros(3)
    src = source.copy()
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        
        # Step 1: Find nearest neighbors
        tree = KDTree(target)
        distances, indices = tree.query(src)
        
        # Step 2: Filter by max distance
        valid = distances < max_correspondence_dist
        if valid.sum() < 10:
            break
        src_matched = src[valid]
        tgt_matched = target[indices[valid]]
        
        # Step 3: Compute optimal R, t via SVD
        src_center = src_matched.mean(axis=0)
        tgt_center = tgt_matched.mean(axis=0)
        src_centered = src_matched - src_center
        tgt_centered = tgt_matched - tgt_center
        
        H = src_centered.T @ tgt_centered  # (3,3) covariance
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection (det(R) must be +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = tgt_center - R @ src_center
        
        # Step 4: Apply transform
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Step 5: Convergence check
        mean_error = distances[valid].mean()
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # Build 4×4 transform matrix
    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3,  3] = t_total
    
    return T, prev_error
```

---

### PHASE 4: REDESIGN ENGINE (Days 5-7)

#### `src/redesign/color_transfer.py`

```python
def reinhard_color_transfer(
    source: np.ndarray,   # furniture image (H, W, 3) BGR
    target: np.ndarray,   # room image (H, W, 3) BGR  
    mask: np.ndarray,     # furniture mask (H, W) uint8
) -> np.ndarray:
    """
    Reinhard et al. (2001) color transfer — implement from scratch.
    Harmonizes new furniture colors with existing room palette.
    
    1. Convert both images BGR → Lab
       src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
       tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    2. For each channel c in [L, a, b]:
       src_mean = src_lab[:,:,c].mean()
       src_std  = src_lab[:,:,c].std()
       tgt_mean = tgt_lab[:,:,c].mean()
       tgt_std  = tgt_lab[:,:,c].std()
       
       result_lab[:,:,c] = (src_lab[:,:,c] - src_mean) / (src_std + 1e-6)
       result_lab[:,:,c] = result_lab[:,:,c] * tgt_std + tgt_mean
    
    3. Clip Lab to valid range, convert back to BGR
    
    4. Apply only within mask region, blend at boundary:
       mask_3ch = mask[:,:,None] / 255.0
       # Feather mask to avoid hard edges
       mask_feathered = cv2.GaussianBlur(mask_3ch, (21, 21), 5)
       result = result * mask_feathered + source * (1 - mask_feathered)
    
    Return result image
    """


def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    Histogram intersection similarity — implement from scratch.
    Both histograms must be normalized (sum to 1).
    Returns value in [0, 1].
    
    return np.minimum(h1, h2).sum()
    """
```

#### `src/redesign/style_engine.py`

```python
class StyleEngine:
    """
    Classical CV style matching — NO neural retrieval, NO embedding models.
    
    Initialization (runs once at startup):
    
    1. Load furniture_index.json
    2. For each item, build feature vector:
       - Load thumbnail image
       - HSV histogram: cv2.calcHist, 32×32×32 bins, normalized
       - Geometric features: [volume, aspect_ratio, height_width_ratio]
       - Style tag, type tag from index
    3. Store all feature vectors in memory
    
    Style color profiles (hardcoded HSV tendencies):
    STYLE_PROFILES = {
        'scandinavian': {'hue_range': [10, 40],  'sat': 'low',  'val': 'high'},
        'industrial':   {'hue_range': [0, 30],   'sat': 'low',  'val': 'low'},
        'bohemian':     {'hue_range': [5, 50],   'sat': 'high', 'val': 'mid'},
        'minimalist':   {'hue_range': None,       'sat': 'low',  'val': 'high'},
        'modern':       {'hue_range': [200, 240], 'sat': 'mid',  'val': 'mid'},
        'traditional':  {'hue_range': [15, 45],   'sat': 'mid',  'val': 'mid'},
    }
    
    def query(self, style: str, furniture_type: str,
              room_color_hist: np.ndarray, top_k: int = 3) -> list[dict]:
        '''
        1. Filter index: type == furniture_type AND style == style
        2. Score each candidate:
           color_score = histogram_intersection(room_color_hist, item.color_hist)
           geo_score   = style_profile_match(item, STYLE_PROFILES[style])
           final_score = 0.6 * color_score + 0.4 * geo_score
        3. Sort descending, return top_k
        '''
    
    def get_room_color_histogram(self, rgb_image: np.ndarray) -> np.ndarray:
        '''
        Compute room's color histogram for matching.
        Resize to 200×200, convert to HSV, compute 32×32×32 hist, normalize.
        '''
    """
```

#### `src/redesign/furniture_placer.py`

```python
def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix R such that R @ v1 = v2.
    Uses Rodrigues rotation formula — implement from scratch.
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(v1, v2)
    if sin_angle < 1e-10:
        return np.eye(3)  # already aligned
    axis = axis / sin_angle
    # Rodrigues formula:
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    return R


class FurniturePlacer:
    """
    Replace old furniture cluster with new furniture point cloud.
    
    def place(self,
              scene_pcd: o3d.PointCloud,
              old_cluster: FurnitureCluster,
              new_furniture_ply_path: str,
              floor_plane: np.ndarray,
              floor_normal: np.ndarray) -> o3d.PointCloud:
        
        1. Load new furniture PCD from .ply
        
        2. SCALE to match old cluster dimensions:
           old_dims = old_cluster.bbox_max - old_cluster.bbox_min
           new_bbox_min = new_pcd.get_min_bound()
           new_bbox_max = new_pcd.get_max_bound()
           new_dims = new_bbox_max - new_bbox_min
           scale = (old_dims / (new_dims + 1e-6))
           # Scale XZ to match footprint, preserve Y proportionally
           new_pcd.scale(scale.mean(), center=new_pcd.get_center())
        
        3. ROTATE to align with floor normal:
           up = np.array([0, 1, 0])
           R = rotation_matrix_from_vectors(up, floor_normal)
           new_pcd.rotate(R, center=new_pcd.get_center())
        
        4. TRANSLATE to old cluster position:
           new_center = new_pcd.get_center()
           target = old_cluster.centroid.copy()
           # Lower until bottom touches floor
           new_bottom = new_pcd.get_min_bound()[1]
           target[1] = floor_plane[3] + (new_center[1] - new_bottom)
           new_pcd.translate(target - new_center)
        
        5. ICP REFINEMENT:
           old_pts = np.asarray(old_cluster.pcd.points)
           new_pts = np.asarray(new_pcd.points)
           T, error = icp_align(new_pts, old_pts, max_iterations=30)
           new_pcd.transform(T)
        
        6. Remove old cluster from scene, merge new furniture
           (use cluster label mask to remove, then add new pcd)
        
        Return modified scene_pcd
    """
```

#### `src/redesign/blend_renderer.py`

```python
class BlendRenderer:
    """
    Project 3D redesigned scene back to 2D frame.
    
    def render_overlay(self,
                       scene_pcd: o3d.PointCloud,
                       rgb_frame: np.ndarray,
                       K: np.ndarray,
                       alpha: float = 0.75) -> np.ndarray:
        '''
        1. Get all points and colors from scene_pcd
           points = np.asarray(scene_pcd.points)   # (N, 3)
           colors = np.asarray(scene_pcd.colors)   # (N, 3) [0,1]
        
        2. Project 3D → 2D (implement from scratch):
           u = (K[0,0] * points[:,0] / points[:,2] + K[0,2]).astype(int)
           v = (K[1,1] * points[:,1] / points[:,2] + K[1,2]).astype(int)
        
        3. Sort by depth (painter's algorithm):
           depth_order = np.argsort(-points[:,2])  # far to near
           u = u[depth_order]
           v = v[depth_order]
           c = (colors[depth_order] * 255).astype(np.uint8)
        
        4. Filter: only valid pixels (0<=u<W, 0<=v<H, Z>0)
        
        5. Create overlay canvas, paint points:
           overlay = np.zeros_like(rgb_frame)
           valid_mask = np.zeros((H, W), dtype=bool)
           overlay[v_valid, u_valid] = c[valid]
           valid_mask[v_valid, u_valid] = True
        
        6. Alpha blend only where overlay has content:
           result = rgb_frame.copy()
           result[valid_mask] = (
               alpha * overlay[valid_mask] +
               (1-alpha) * rgb_frame[valid_mask]
           ).astype(np.uint8)
        
        Return result frame
        '''
    """
```

---

### PHASE 5: REAL-TIME PIPELINE (Days 8-11) — THE STAR

#### `src/realtime/temporal_smoother.py`

```python
class KalmanSmoother3D:
    """
    3D Kalman filter for furniture tracking — implement from scratch.
    One instance per tracked furniture cluster.
    
    State: [x, y, z, vx, vy, vz]  shape (6,)
    Observation: [x, y, z]         shape (3,)
    
    def __init__(self, initial_position: np.ndarray, dt: float = 1/6):
        # State: position + velocity
        self.x = np.array([*initial_position, 0, 0, 0], dtype=float)
        
        # State transition (constant velocity)
        self.F = np.eye(6)
        self.F[0,3] = self.F[1,4] = self.F[2,5] = dt
        
        # Observation matrix (we observe position only)
        self.H = np.zeros((3, 6))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = 1.0
        
        # Covariance matrices
        self.P = np.eye(6) * 0.1        # state covariance
        self.Q = np.eye(6) * 0.001      # process noise
        self.R = np.eye(3) * 0.05       # measurement noise
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        y = measurement - self.H @ self.x          # innovation
        S = self.H @ self.P @ self.H.T + self.R    # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.x[:3]
    
    def smooth(self, measurement: np.ndarray) -> np.ndarray:
        self.predict()
        return self.update(measurement)
    """
```

#### `src/realtime/pipeline.py`

```python
class RealTimePipeline:
    """
    3-thread architecture for smooth real-time AR.
    The illusion of real-time: fast CV runs every frame,
    heavy processing runs async and replaces overlay when ready.
    
    LATENCY BUDGET (1 second total target):
    - Frame capture:        0ms  (hardware)
    - Depth inference:     40ms  (thread 2, every 5th frame)
    - Mask warp (optical): 10ms  (thread 2, every frame)
    - Style color xfer:     5ms  (thread 2, every frame)
    - Overlay blend:        8ms  (thread 3, every frame)
    - Heavy inpaint (SD):  ~3s   (async, user doesn't see wait)
    
    ─────────────────────────────────────────────
    THREAD 1 — CaptureThread (30 fps, daemon)
    ─────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_buffer = collections.deque(maxlen=5)
    Just captures and pushes to buffer. Never blocks.
    
    ─────────────────────────────────────────────
    THREAD 2 — ProcessThread (adaptive)
    ─────────────────────────────────────────────
    frame_counter = 0
    
    Every frame (30fps):
        # Warp cached segmentation mask using optical flow
        if prev_frame is not None and cached_mask is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            cached_mask = warp_mask_with_flow(cached_mask, flow)
        
        # Apply color transfer to masked region (instant)
        if target_style is not None:
            apply_realtime_color_transfer(frame, cached_mask, target_style)
    
    Every 5th frame (~6fps):
        # Run FastDepthNet on downscaled frame
        depth = fast_depth_net.infer(resize(frame, 256, 256))
        update depth_cache
        
        # Back-project → update point cloud cache
        pcd = depth_to_pointcloud(frame, depth, K)
        pcd = clean_pointcloud(pcd, voxel_size=0.04)  # fast, coarse
        update pcd_cache
    
    Every 15th frame (~2fps):
        # Run SAM segmentation
        masks = sam_model.segment(resize(frame, 512, 512))
        update seg_cache
    
    Every 30th frame (~1fps):
        # Full RANSAC + DBSCAN
        geometry = detect_room_geometry(pcd_cache)
        clusters = classify_clusters(dbscan_3d(geometry.furniture_pts))
        update cluster_cache
        
        # Apply Kalman smoothing to cluster positions
        for cluster in clusters:
            if cluster.label in kalman_trackers:
                pos = kalman_trackers[cluster.label].smooth(cluster.centroid)
                cluster.centroid = pos
            else:
                kalman_trackers[cluster.label] = KalmanSmoother3D(cluster.centroid)
        
        # If pending style command: run furniture swap
        if pending_command:
            new_scene = redesign_pipeline.execute(pending_command)
            update overlay_cache
            pending_command = None
    
    ─────────────────────────────────────────────
    THREAD 3 — RenderThread (30 fps)
    ─────────────────────────────────────────────
    Always displays. Never blocks.
    
    Every frame:
        frame = frame_buffer[-1]
        overlay = overlay_cache  # last computed redesign
        
        if overlay is not None:
            result = alpha_blend(frame, overlay, alpha=0.75)
        else:
            result = frame.copy()
        
        # Draw UI elements (depth map thumbnail, FPS, labels)
        draw_ui_overlay(result, fps, current_style, cluster_cache)
        cv2.imshow('RoomAI', result)
    
    Thread communication:
        Use threading.Lock() for: depth_cache, pcd_cache, seg_cache,
                                   cluster_cache, overlay_cache
        frame_buffer is collections.deque — thread-safe by default
    """
```

#### `src/ui/opencv_window.py`

```python
class RoomAIDisplay:
    """
    Full OpenCV window — 1280×720 layout.
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │                  MAIN VIEW  960×720                             │
    │              [Live + AR overlay]                                │
    │                                                                 │
    └─────────────────────────────────────────┬───────────────────────┘
                                              │  SIDEBAR  320×720
    ┌─────────────────────────────────────────┤
    │ 🏠 RoomAI    FPS: 28    Mode: Scandi    │
    ├─────────────────────────────────────────┤
    │ STYLE                                   │
    │ [1] Modern     [2] Scandinavian         │
    │ [3] Industrial [4] Bohemian             │
    │ [5] Minimalist [6] Traditional          │
    ├─────────────────────────────────────────┤
    │ VIEW                                    │
    │ [D] Depth   [S] Segments                │
    │ [P] PCloud  [O] Original                │
    │ [R] Redesign [B] Before/After           │
    ├─────────────────────────────────────────┤
    │ ROOMS                                   │
    │ > Living Room  [N] Next [←][→] prev    │
    │   Bedroom                               │
    │   Kitchen                               │
    ├─────────────────────────────────────────┤
    │ Pipeline Status                         │
    │ Depth:    ✅ 42ms                       │
    │ Segments: ✅ 4 clusters                 │
    │ Voice:    🎤 listening                  │
    │ Style:    Scandinavian                  │
    └─────────────────────────────────────────┘
    
    Keyboard shortcuts:
    1-6: switch style
    d:   depth overlay
    s:   segmentation overlay
    p:   point cloud overlay (open3d visualizer)
    o:   original (no overlay)
    r:   redesigned view
    b:   before/after split slider
    n:   next room
    space: pause/resume processing
    q:   quit and save session
    
    Mouse click on furniture cluster: highlight + show tooltip
    {type, volume, style_match, dimensions}
    """
```

---

### PHASE 6: VOICE COMMANDS (Day 12)

#### `src/ui/voice_commands.py`

```python
import speech_recognition as sr

STYLE_COMMANDS = {
    'scandinavian': 'scandinavian', 'scandi': 'scandinavian',
    'nordic': 'scandinavian', 'industrial': 'industrial',
    'minimalist': 'minimalist', 'minimal': 'minimalist',
    'bohemian': 'bohemian', 'boho': 'bohemian',
    'modern': 'modern', 'contemporary': 'modern',
    'traditional': 'traditional', 'classic': 'traditional',
}

ACTION_COMMANDS = {
    'remove sofa': ('remove', 'sofa'),
    'remove chair': ('remove', 'chair'),
    'remove table': ('remove', 'table'),
    'add plant': ('add', 'plant'),
    'add lamp': ('add', 'lamp'),
    'reset room': ('reset', None),
    'reset': ('reset', None),
}

NAV_COMMANDS = {
    'next room': 'next', 'next': 'next',
    'bedroom': 'bedroom', 'living room': 'living_room',
    'kitchen': 'kitchen', 'bathroom': 'bathroom',
}

VIEW_COMMANDS = {
    'show depth': 'depth', 'depth': 'depth',
    'show point cloud': 'pointcloud', 'point cloud': 'pointcloud',
    'show original': 'original', 'original': 'original',
    'before after': 'split', 'compare': 'split',
    'show redesign': 'redesign',
}

class VoiceCommandHandler:
    """
    Runs in daemon thread. Listens continuously.
    On recognized command, pushes to command_queue.
    
    Uses Google STT via SpeechRecognition (sr.Recognizer).
    Falls back to sr.recognize_sphinx for offline use.
    
    def listen_loop(self, command_queue: queue.Queue):
        r = sr.Recognizer()
        r.energy_threshold = 3000
        r.dynamic_energy_threshold = True
        mic = sr.Microphone()
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=1)
        while True:
            with mic as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=4)
            try:
                text = r.recognize_google(audio).lower()
                cmd = parse_command(text)
                if cmd:
                    command_queue.put(cmd)
            except (sr.UnknownValueError, sr.RequestError):
                pass
    """
```

---

### PHASE 7: MULTI-ROOM GRAPH (Days 9-10)

#### `src/scene/room_graph.py`

```python
@dataclass
class RoomNode:
    room_id: str
    room_type: str              # living_room, bedroom, kitchen, etc.
    current_style: str
    original_image: np.ndarray
    redesigned_image: np.ndarray
    depth_map: np.ndarray
    pcd: o3d.geometry.PointCloud
    clusters: list[FurnitureCluster]
    floor_plane: np.ndarray     # (4,) [a,b,c,d]
    wall_planes: list
    floor_plan_2d: np.ndarray   # top-down projection
    blip_metadata: dict         # from metadata.jsonl

class RoomGraph:
    """
    Graph of rooms for whole-house redesign.
    nodes: dict[room_id, RoomNode]
    edges: list[tuple[room_id, room_id]]  # connected rooms
    
    def add_room_from_image(self, image_path: str, room_id: str = None):
        Run full single-room pipeline → RoomNode → add to graph
    
    def apply_style_to_all(self, style: str):
        For each room:
            run redesign pipeline with this style
            maintain consistent color palette
        Use BFS from root room to propagate style
    
    def generate_floor_plan(self) -> np.ndarray:
        Top-down 2D floor plan from all room point clouds.
        For each room:
            Project points onto XZ plane → 2D occupancy grid
            Grid resolution: 0.05m per pixel
            Find floor boundaries (convex hull of floor cluster)
        Stitch all rooms together
        Draw: floor outline, furniture footprints, room labels
        Return (H, W, 3) floor plan image — good for demo
    
    def get_graph_json(self) -> dict:
        Return serializable dict for frontend:
        {rooms: [{id, type, style, thumbnail_url, cluster_count}],
         edges: [[room_a, room_b]]}
    """
```

---

### PHASE 8: ML TRAINING (Days 3-5 overnight, 13-15 overnight)

#### `ml/training/finetune_sd_lora.py`

```python
# LoRA fine-tune SD 1.5 on our 3,826 room images
# Run overnight: nohup python ml/training/finetune_sd_lora.py > lora.log 2>&1 &

CONFIG = {
    "model_id":                   "runwayml/stable-diffusion-v1-5",
    "device":                     "mps",
    "lora_rank":                  4,
    "lora_alpha":                 8,
    "target_modules":             ["to_q", "to_v", "to_k", "to_out.0"],
    "batch_size":                 1,
    "gradient_accumulation_steps": 8,    # effective batch = 8
    "learning_rate":              1e-4,
    "max_train_steps":            3000,
    "mixed_precision":            "no",  # MPS fp16 training unstable
    "resolution":                 512,
    "checkpoint_every":           500,
    "validate_every":             500,
    "n_validation_samples":       4,
    "wandb":                      True,
}

# Prompt template for each image:
# "{style} {room_type} interior design, {color_0} and {color_1} palette,
#  {objects}, professional interior photography, high quality, 8k"

# Training loop:
# - torch.mps.empty_cache() every 100 steps
# - gradient checkpointing: unet.enable_gradient_checkpointing()
# - Save: ml/checkpoints/lora_step{N}.safetensors
# - Log loss + sample images to wandb every validate_every steps
# Estimated time: 18-22 hours on M4
```

#### `ml/training/train_controlnet.py`

```python
# Train ControlNet conditioned on MiDaS depth maps
# Run overnight after LoRA done

CONFIG = {
    "base_model":                 "runwayml/stable-diffusion-v1-5",
    "device":                     "mps",
    "batch_size":                 1,
    "gradient_accumulation":      16,
    "learning_rate":              1e-5,
    "max_steps":                  5000,
    "resolution":                 512,
    "mixed_precision":            "no",
}

# Data preparation:
# For each training image:
#   1. Run MiDaS → depth map (384×384)
#   2. Normalize: (d - d.min()) / (d.max() - d.min())
#   3. Make 3-channel: np.stack([d_norm]*3, axis=-1) * 255
#   Training pair: (depth_3ch, room_rgb, caption)

# Architecture: diffusers ControlNetModel
# Base UNet: frozen
# ControlNet: trainable encoder copy
# Save: ml/checkpoints/controlnet_depth_step{N}/
# Estimated time: 24-30 hours on M4
```

---

### PHASE 9: SEGMENTATION MODEL (Days 2-3, parallel with depth)

#### `src/segmentation/unet.py`

```python
class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""

class Down(nn.Module):
    """MaxPool2d(2) → DoubleConv"""

class Up(nn.Module):
    """Bilinear upsample × 2 → concat skip → DoubleConv"""

class UNet(nn.Module):
    """
    UNet from scratch — 10 indoor semantic classes.
    
    Classes: floor, wall, ceiling, sofa, chair, table,
             cabinet, lamp, plant, other
    
    Architecture:
    Encoder:
      inc:  DoubleConv(3, 64)          → 384×384
      down1: Down(64, 128)             → 192×192
      down2: Down(128, 256)            → 96×96
      down3: Down(256, 512)            → 48×48
      down4: Down(512, 1024)           → 24×24
    
    Decoder:
      up1: Up(1024+512, 512)           → 48×48
      up2: Up(512+256, 256)            → 96×96
      up3: Up(256+128, 128)            → 192×192
      up4: Up(128+64, 64)              → 384×384
      outc: Conv2d(64, 10, kernel_size=1)
    
    Loss: CrossEntropyLoss with class weights
    (compute weights from class frequency in ADE20K)
    
    Input:  (B, 3, 384, 384)
    Output: (B, 10, 384, 384) — logits
    """
```

#### `src/segmentation/train_seg.py`

```python
# Train UNet on ADE20K indoor subset
# Dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/
# Use: images in data/ade20k_indoor/ with annotations

CONFIG = {
    "device":          "mps",
    "batch_size":      4,
    "lr":              1e-4,
    "epochs":          50,
    "input_size":      (384, 384),
    "n_classes":       10,
    "checkpoint_path": "models/seg_model.pth",
}

# Augmentation (albumentations):
# HorizontalFlip(p=0.5)
# RandomBrightnessContrast(p=0.3)
# ShiftScaleRotate(shift=0.1, scale=0.1, rotate=15, p=0.5)
# Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# Training: ~6-8 hours on M4
# Target: mIoU > 0.65 on ADE20K indoor val set
```

---

### PHASE 10: EVALUATION (Day 14)

#### `src/evaluation/metrics.py`

```python
# ALL metrics implemented from scratch — NumPy only

def compute_depth_metrics(pred, gt):
    """delta1, delta2, delta3, rmse, abs_rel, sq_rel, log_rms"""

def compute_ssim(img1, img2, window_size=11):
    """
    SSIM from scratch. Sliding window convolution.
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    kernel = gaussian_kernel(window_size, sigma=1.5)
    
    mu1 = convolve(img1, kernel)
    mu2 = convolve(img2, kernel)
    mu1_sq  = mu1**2
    mu2_sq  = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve(img1**2, kernel) - mu1_sq
    sigma2_sq = convolve(img2**2, kernel) - mu2_sq
    sigma12   = convolve(img1*img2, kernel) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
               ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
    """

def compute_segmentation_metrics(pred_mask, gt_mask, n_classes=10):
    """mIoU per class, pixel accuracy — from scratch"""

def compute_fid(real_dir, gen_dir):
    """
    FID using InceptionV3 features. Frechet distance from scratch.
    μ_r, Σ_r = mean and cov of real features
    μ_g, Σ_g = mean and cov of generated features
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r·Σ_g)^0.5)
    Use scipy.linalg.sqrtm for matrix square root.
    """

def compute_clip_score(image: np.ndarray, text: str) -> float:
    """Load CLIP, encode image + text, return cosine similarity"""

def run_full_evaluation(test_set_path, results_dir) -> dict:
    """
    Run all metrics on test set.
    Output:
    - evaluation_report.json
    - latex_table.txt  (copy-paste into report)
    
    Table format:
    Method | delta1 | RMSE | mIoU | SSIM | FID | CLIP | FPS
    Ours   | 0.XX   | X.XX | X.XX | X.XX | XX  | X.XX | 28
    """
```

---

### PHASE 11: BACKEND API UPDATES

#### New endpoints (add to existing FastAPI app):

```python
# backend/app/routes/redesign.py

@router.post("/api/redesign/realtime")
async def redesign_realtime(frame_data: FrameRequest):
    """
    Main real-time endpoint.
    Input: base64 frame + style command + session_id
    Output: {overlay_frame: base64, clusters: [...], depth_thumbnail: base64}
    Target latency: < 300ms (color transfer + mask warp only)
    """

@router.post("/api/redesign/full")
async def redesign_full(image: UploadFile, style: str, room_id: str):
    """
    Full inpainting redesign (async job).
    Returns job_id immediately.
    Use Celery task: run SD inpainting → store result → notify via SSE
    """

@router.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    SSE stream for redesign job progress.
    Events: {status: 'processing', stage: 'depth|seg|inpaint|upscale', progress: 0.0-1.0}
    """

@router.post("/api/pointcloud/generate")
async def generate_pointcloud(image_id: str):
    """
    Input: image_id
    Output: {ply_url, floor_plane, wall_planes, clusters: [...]}
    """

@router.post("/api/rooms/add")
async def add_room(image: UploadFile, room_type: str):
    """Input: image → full room analysis → add to session graph"""

@router.get("/api/rooms/graph")
async def get_room_graph():
    """Return full room graph for frontend visualization"""

@router.post("/api/rooms/style-all")
async def style_all_rooms(style: str, background_tasks: BackgroundTasks):
    """Apply style to all rooms in graph asynchronously"""
```

---

### PHASE 12: FRONTEND COMPONENTS

#### `frontend/src/components/RealtimeViewer.tsx`
```
Webcam → captures frames → sends to /api/redesign/realtime
Displays returned overlay_frame
Overlays cluster bounding boxes
Shows FPS counter
Voice activity indicator (animated mic icon when listening)
```

#### `frontend/src/components/BeforeAfterSlider.tsx`
```
Draggable vertical divider
Left: original image
Right: redesigned image
Drag to reveal before/after
```

#### `frontend/src/components/PointCloudViewer.tsx`
```
Load .ply file from ply_url
Display using Three.js Points geometry
Orbit controls (rotate, zoom)
Color by: RGB / depth / cluster label (toggle)
```

#### `frontend/src/components/FloorPlan.tsx`
```
Display floor plan PNG generated by room_graph
Clickable rooms (highlight selected)
Room labels + furniture footprint outlines
```

#### `frontend/src/components/EvaluationDashboard.tsx`
```
Display metrics table from evaluation_report.json
delta1/2/3, RMSE, mIoU, SSIM, FID, CLIP Score, FPS
Bar charts for visual comparison
```

---

## ALGORITHMS IMPLEMENTED FROM SCRATCH (exam requirement)

All of these must be pure NumPy/Python — no library shortcuts:

1. **RANSAC plane fitting** — `src/geometry/ransac_plane.py`
2. **DBSCAN clustering** — `src/geometry/dbscan_custom.py`
3. **Depth back-projection** — `src/pointcloud/backprojector.py`
4. **BerHu loss** — `src/depth/fast_depth_net.py`
5. **UNet architecture** — `src/segmentation/unet.py`
6. **3D Kalman filter** — `src/realtime/temporal_smoother.py`
7. **Reinhard color transfer** — `src/redesign/color_transfer.py`
8. **ICP alignment** — `src/geometry/icp.py`
9. **PCA normal estimation** — `src/pointcloud/normal_estimator.py`
10. **Depth metrics** (delta1/2/3, RMSE, AbsRel, SqRel) — `src/depth/depth_evaluator.py`
11. **SSIM** — `src/evaluation/metrics.py`
12. **Histogram intersection** — `src/redesign/color_transfer.py`
13. **Rodrigues rotation formula** — `src/redesign/furniture_placer.py`

---

## DATASETS

```
NYU Depth v2:
  URL: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
  Format: HDF5, 1449 images, 795 train / 654 test
  Use for: FastDepthNet training + depth metric evaluation

ADE20K Indoor Subset:
  URL: https://groups.csail.mit.edu/vision/datasets/ADE20K/
  Filter: images in scene categories: bedroom, living_room, kitchen, bathroom
  Use ~10k images, 10 semantic classes
  Use for: UNet segmentation training

Our Dataset (already built):
  3,826 images (IKEA + Unsplash + House)
  data/annotations/metadata.jsonl
  Use for: LoRA fine-tuning, ControlNet training, style engine
```

---

## MAC M4 TRAINING SCHEDULE

```
Day 1 (afternoon):
  python src/depth/train_depth.py
  Duration: ~4 hours | Watch: validation delta1 > 0.85

Night 1-2:
  nohup python ml/training/finetune_sd_lora.py > lora.log 2>&1 &
  Duration: ~20 hours | Watch: loss curve in wandb

Night 3 (after ControlNet data prep):
  nohup python ml/training/train_controlnet.py > controlnet.log 2>&1 &
  Duration: ~26 hours | Watch: validation samples in wandb

Parallel (anytime):
  python src/segmentation/train_seg.py
  Duration: ~8 hours | Watch: mIoU on ADE20K val
```

---

## 30-DAY SCHEDULE

```
Day 1-2:    Dataset finalizer → furniture PCD library → FastDepthNet training
Day 3:      Backprojector → cleaner → normal estimator (full PCD pipeline)
Day 4:      RANSAC → DBSCAN → furniture classifier
Day 5:      ICP → furniture placer → blend renderer
Day 6:      Style engine → color transfer → end-to-end single image redesign
Day 7:      Single image demo working — test and fix
Day 8-9:    Real-time 3-thread pipeline — capture/process/render
Day 10:     Kalman filter → temporal smoothing → stable overlay
Day 11:     Voice commands → keyboard shortcuts → full OpenCV UI
Day 12:     Multi-room graph → floor plan generator
Day 13:     Start ControlNet training overnight
Day 14:     UNet segmentation training → evaluation metrics
Day 15-16:  Frontend: RealtimeViewer, BeforeAfterSlider, PointCloudViewer
Day 17:     Backend API endpoints → connect frontend
Day 18-20:  Full end-to-end integration testing
Day 21-22:  Demo polish — rehearse 3-minute demo
Day 23-25:  Report writing + evaluation table
Day 26-27:  Jupyter notebooks (01-06)
Day 28-30:  Buffer — edge cases, final submission
```

---

## PERFORMANCE TARGETS

```
Real-time:
  Display FPS:          30 fps (render thread)
  Processing FPS:        6 fps (depth + mask warp)
  Full update FPS:       1 fps (RANSAC + DBSCAN)
  Style command latency: < 1 second (color transfer, instant)
  Full redesign latency: 3-5 seconds (SD inpainting, async)

Depth estimation:
  delta1 > 0.85
  RMSE < 0.55m
  AbsRel < 0.14

Segmentation:
  mIoU > 0.65
  Pixel accuracy > 0.80

Redesign quality:
  SSIM > 0.85 (structure preserved)
  CLIP Score > 0.28 (style alignment)
  FID < 40 (realism)
```

---

## DEMO SCRIPT (3 minutes, professor-facing)

```
0:00 — Upload 3 room photos → system builds room graph
0:15 — Floor plan appears on screen showing connected rooms
0:25 — Click living room → Studio view opens
0:35 — Press [D] — show depth heatmap
0:40 — Press [S] — show colored segmentation masks (4 furniture clusters)
0:45 — Press [P] — 3D point cloud viewer opens, orbits room
0:55 — Press [R] — switch to redesigned view
1:00 — Say "industrial" → overlay switches to industrial style live
1:08 — Drag before/after slider — dramatic reveal
1:15 — Press [N] → next room (bedroom) → different redesign shown
1:22 — Say "make the whole house scandinavian" → all 3 rooms redesign
1:35 — Switch to WEBCAM MODE → point at real desk/room
1:40 — Overlay appears on live feed
1:50 — Say "minimalist" → color transfer applies instantly
2:00 — Show sidebar: FPS counter, cluster labels, voice indicator
2:10 — Show evaluation dashboard: delta1, mIoU, SSIM table
2:25 — Explain pipeline architecture on prepared diagram slide
3:00 — Done
```

---

## NON-NEGOTIABLES

1. No external AI APIs in CV pipeline. BLIP-2 was used for annotation only.
2. All 13 from-scratch algorithms must be pure NumPy/Python.
3. Real-time overlay must feel smooth (30fps display, <1s style switch).
4. Every file must be complete and runnable. No TODOs. No stubs.
5. MPS device used for all inference. `torch.mps.empty_cache()` after heavy ops.
6. Training runs overnight unattended via nohup.
7. Final demo must work offline (no internet dependency at runtime).

---

## START HERE

Build in this exact order. Each phase depends on the previous.

Phase 1 first → run depth training in background → Phase 2 while training.

The 1-second style switch latency is the north star.
Every decision you make should ask: does this help or hurt that target?
```
