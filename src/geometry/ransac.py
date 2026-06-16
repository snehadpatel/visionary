import numpy as np

def fit_plane_ransac(
    points: np.ndarray,   # (N, 3)
    threshold: float = 0.015,
    n_iter: int = 1000,
    min_inliers: int = 100,
) -> tuple[np.ndarray or None, np.ndarray]:
    """
    Pure NumPy RANSAC for plane fitting.
    Returns: (plane [a,b,c,d] or None, inlier_mask)
    """
    N = len(points)
    if N < 3:
        return None, np.zeros(N, bool)
        
    best_n = 0
    best_plane = None
    best_mask = np.zeros(N, bool)
    
    for _ in range(n_iter):
        # 1. Sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # 2. Compute plane normal
        normal = np.cross(p2 - p1, p3 - p1)
        mag = np.linalg.norm(normal)
        if mag < 1e-10: continue
        normal /= mag
        
        # 3. Compute D parameter (ax + by + cz + d = 0)
        d = -np.dot(normal, p1)
        
        # 4. Find inliers
        dists = np.abs(np.dot(points, normal) + d)
        mask = dists < threshold
        n = mask.sum()
        
        if n > best_n:
            best_n = n
            best_plane = np.append(normal, d)
            best_mask = mask
            
    if best_n < min_inliers:
        return None, np.zeros(N, bool)
        
    # 5. Refit with all inliers via SVD for better accuracy
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    centered = inliers - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1] # smallest eigenvalue
    # Ensure standard orientation (usually Y-up for floor)
    if normal[1] < 0: normal = -normal
    d = -np.dot(normal, centroid)
    
    return np.append(normal, d), best_mask

def detect_room_planes(points: np.ndarray) -> dict:
    """
    Iteratively find floor then walls.
    """
    remaining = points.copy()
    remaining_idx = np.arange(len(points))
    
    result = {
        'floor_plane': None,
        'wall_planes': [],
        'floor_height': 0.0,
        'floor_normal': np.array([0., 1., 0.]),
        'furniture_mask': np.ones(len(points), bool)
    }
    
    # 1. Floor Detection (horizontal normal |Ny| > 0.7)
    plane, mask = fit_plane_ransac(remaining)
    floor_mask_all = np.zeros(len(points), bool)
    if plane is not None and abs(plane[1]) > 0.7:
        result['floor_plane'] = plane
        result['floor_normal'] = plane[:3]
        result['floor_height'] = remaining[mask, 1].mean()
        
        floor_mask_all[remaining_idx[mask]] = True
        
        # Remove floor points
        remaining = remaining[~mask]
        remaining_idx = remaining_idx[~mask]
    result['floor_mask'] = floor_mask_all
        
    # 1.5 Ceiling Detection (Ny < -0.85)
    ceiling_mask_all = np.zeros(len(points), bool)
    plane, mask = fit_plane_ransac(remaining)
    if plane is not None and plane[1] < -0.85:
        ceiling_mask_all[remaining_idx[mask]] = True
        # Remove ceiling points
        remaining = remaining[~mask]
        remaining_idx = remaining_idx[~mask]
    result['ceiling_mask'] = ceiling_mask_all
        
    # 2. Wall Detection (vertical normals |Ny| < 0.3)
    wall_mask_all = np.zeros(len(points), bool)
    for _ in range(4): # Find up to 4 walls
        if len(remaining) < 100: break
        plane, mask = fit_plane_ransac(remaining)
        if plane is not None and abs(plane[1]) < 0.3:
            result['wall_planes'].append(plane)
            wall_mask_all[remaining_idx[mask]] = True
            remaining = remaining[~mask]
            remaining_idx = remaining_idx[~mask]
        else:
            break
    result['wall_mask'] = wall_mask_all
            
    # Everything left is furniture
    furniture_mask = np.zeros(len(points), bool)
    furniture_mask[remaining_idx] = True
    result['furniture_mask'] = furniture_mask
    
    return result

if __name__ == "__main__":
    # Dummy test
    points = np.random.randn(1000, 3)
    # Simulate a floor at y=0
    points[:500, 1] = np.random.normal(0, 0.01, 500)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    room = detect_room_planes(pcd)
    print(f"Floor detected: {room['floor_plane'] is not None}")
    if room['floor_plane'] is not None:
        print(f"Floor height: {room['floor_height']:.4f}")
    print(f"Walls detected: {len(room['wall_planes'])}")
