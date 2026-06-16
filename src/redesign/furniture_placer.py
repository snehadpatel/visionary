import numpy as np
import cv2
import struct
import os

class FurniturePlacer:
    """
    Handles 3D asset loading (.ply), spatial transformation, 
    auto-layout heuristics, and 2D reprojection (Depth/Mask).
    """
    def __init__(self, library_index: list):
        self.library = library_index
        
    def load_ply_binary(self, filepath: str):
        """
        Fast binary PLY loader for files with (x,y,z,r,g,b).
        """
        if not os.path.exists(filepath):
            return None, None
            
        with open(filepath, 'rb') as f:
            header = ""
            while "end_header" not in header:
                header += f.readline().decode('ascii')
            
            # Find vertex count
            import re
            match = re.search(r"element vertex (\d+)", header)
            if not match: return None, None
            num_verts = int(match.group(1))
            
            # Load data: float32[3] + uint8[3]
            # dtype matches backprojector.py exporter
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
            data = np.fromfile(f, dtype=dtype, count=num_verts)
            
            points = np.stack([data['x'], data['y'], data['z']], axis=-1)
            colors = np.stack([data['r'], data['g'], data['b']], axis=-1)
            
            return points, colors

    def get_transform_matrix(self, translation=None, rotation_y=0, scale=1.0):
        """Create a 4x4 transformation matrix."""
        T = np.eye(4)
        if translation is not None:
            T[:3, 3] = translation
        
        # Rotation around Y axis (vertical)
        rad = np.radians(rotation_y)
        R_y = np.array([
            [np.cos(rad),  0, np.sin(rad)],
            [0,            1, 0          ],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        T[:3, :3] = R_y * scale
        return T

    def apply_transform(self, points, T):
        """Apply 4x4 transform to points."""
        if points is None or len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T @ points_homo.T).T
        return transformed[:, :3]

    def suggest_layout_heuristic(self, room_geometry: dict, target_style: str):
        """
        Auto-Layout Logic:
        1. Identify Primary Wall (longest one)
        2. Orient Sofa parallel to it.
        3. Place Coffee Table in front.
        """
        floor_h = room_geometry.get('floor_height', 0.0)
        wall_planes = room_geometry.get('wall_planes', [])
        
        if not wall_planes:
            # Default fallback if no walls detected
            return [
                {'type': 'sofa', 'pos': [0, floor_h, 3.0], 'rot': 0},
                {'type': 'table', 'pos': [0, floor_h, 2.2], 'rot': 0}
            ]
            
        # 1. Primary Wall: Largest wall
        primary_wall = wall_planes[0] 
        normal = primary_wall[:3]
        
        # Calculate angle of the wall normal on the XZ plane to orient furniture
        angle_rad = np.arctan2(normal[0], normal[2])
        angle_deg = np.degrees(angle_rad)
        
        # 2. Place Sofa 0.2m off the wall
        sofa_pos = np.array([0.0, floor_h, 3.5]) 
        
        return [
            {'type': 'sofa',  'pos': sofa_pos, 'rot': 180 + angle_deg},
            {'type': 'table', 'pos': sofa_pos - np.array([0, 0, 1.0]), 'rot': angle_deg}
        ]

    def synthesize_scene(self, assets: list, K, h, w):
        """
        Render multiple 3D assets into a single Depth map and Mask.
        Vectorized Z-buffer using NumPy.
        """
        full_depth = np.zeros((h, w), dtype=np.float32)
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        all_u = []
        all_v = []
        all_z = []
        
        for asset in assets:
            pts = self.apply_transform(asset['points'], asset['transform'])
            if len(pts) == 0: continue
            
            # Project
            pts_img = (K @ pts.T).T
            
            # Avoid division by zero
            z = pts_img[:, 2]
            valid_z = np.abs(z) > 1e-6
            
            u = (pts_img[valid_z, 0] / z[valid_z]).astype(int)
            v = (pts_img[valid_z, 1] / z[valid_z]).astype(int)
            z = z[valid_z]
            
            # Filter bounds
            in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0.1)
            all_u.append(u[in_view])
            all_v.append(v[in_view])
            all_z.append(z[in_view])
            
        if not all_u:
            return full_depth, full_mask
            
        # Concatenate all points
        u = np.concatenate(all_u)
        v = np.concatenate(all_v)
        z = np.concatenate(all_z)
        
        # Vectorized Z-buffer:
        # Sort by depth descending (farthest first)
        # When we assign to the array, the nearest points (last in list) will win.
        idx = np.argsort(z)[::-1]
        full_depth[v[idx], u[idx]] = z[idx]
        full_mask[v[idx], u[idx]] = 255
        
        # Post-process mask (fill small holes)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        return full_depth, full_mask

