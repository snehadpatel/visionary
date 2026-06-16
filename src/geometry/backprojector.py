import numpy as np
from scipy.spatial import KDTree

def estimate_intrinsics(W: int, H: int, fov_deg: float = 65.0) -> np.ndarray:
    fx = W / (2 * np.tan(np.radians(fov_deg) / 2))
    fy = fx * H / W
    cx, cy = W / 2.0, H / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

def save_ply(filename, points, colors=None):
    """
    Binary PLY exporter.
    """
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
    if colors is not None:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        if colors is not None:
            # Interleave x, y, z, r, g, b
            data = np.empty(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
            data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
            data['r'], data['g'], data['b'] = colors[:, 0], colors[:, 1], colors[:, 2]
            f.write(data.tobytes())
        else:
            data = np.empty(len(points), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
            f.write(data.tobytes())

def depth_to_pointcloud_np(rgb, depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth.astype(np.float32)
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    
    valid = (Z > 0.1) & (Z < 10.0)
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    colors = rgb[valid].astype(np.uint8)
    return points, colors

def estimate_normals_np(points, radius=0.1, max_nn=30):
    tree = KDTree(points)
    normals = []
    for i, pt in enumerate(points):
        idxs = tree.query_ball_point(pt, radius)
        if len(idxs) < 3:
            normals.append([0.0, 1.0, 0.0])
            continue
        nbrs = points[idxs[:max_nn]]
        centered = nbrs - nbrs.mean(axis=0)
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        if normal[1] < 0: normal = -normal # Simple orientation
        normals.append(normal)
    return np.array(normals)

def clean_pointcloud_np(points, colors, voxel_size=0.02):
    """
    Grid-based downsampling.
    """
    if len(points) == 0: return points, colors
    coords = (points / voxel_size).astype(int)
    # Simple voxel downsampling via dict
    voxels = {}
    for i in range(len(points)):
        key = tuple(coords[i])
        if key not in voxels:
            voxels[key] = i
    idx = list(voxels.values())
    return points[idx], colors[idx]
