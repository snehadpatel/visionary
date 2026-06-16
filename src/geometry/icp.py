import numpy as np
from scipy.spatial import KDTree

def icp_align(
    source: np.ndarray,   # (N, 3) points to transform
    target: np.ndarray,   # (M, 3) reference points
    max_iter: int = 50,
    tol: float = 1e-6,
    max_dist: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    ICP (Iterative Closest Point) alignment implementation from scratch.
    """
    N = len(source)
    M = len(target)
    
    if N == 0 or M == 0:
        return np.eye(4), 0.0
        
    R_total = np.eye(3)
    t_total = np.zeros(3)
    src = source.copy()
    prev_err = np.inf
    
    tree = KDTree(target)
    
    for i in range(max_iter):
        # 1. Find correspondence
        dists, idx = tree.query(src)
        
        valid = dists < max_dist
        if valid.sum() < 6: # Need enough points for SVD
            break
            
        s = src[valid]
        t = target[idx[valid]]
        
        # 2. Compute centroids
        sc = s.mean(axis=0)
        tc = t.mean(axis=0)
        
        # 3. Compute cross-covariance matrix
        # H = (s - sc).T @ (t - tc)
        H = np.dot((s - sc).T, (t - tc))
        
        # 4. SVD for rotation
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = np.dot(Vt.T, U.T)
            
        # 5. Compute translation
        t_vec = tc - np.dot(R, sc)
        
        # 6. Update points and total transformation
        src = np.dot(R, src.T).T + t_vec
        R_total = np.dot(R, R_total)
        t_total = np.dot(R, t_total) + t_vec
        
        # 7. Check convergence
        err = dists[valid].mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err
        
    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = t_total
    
    return T, prev_err

if __name__ == "__main__":
    # Dummy test
    source = np.random.randn(100, 3)
    # Apply a known rotation and translation
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) # 90 deg Z
    t = np.array([1, 2, 3])
    target = np.dot(R, source.T).T + t
    
    T, err = icp_align(source, target)
    print(f"ICP Error: {err:.6f}")
    print("Estimated T:")
    print(T)
