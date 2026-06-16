import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

def extract_3d_objects(result_img, depth_map, masks, K, output_dir, client_id):
    """
    Extracts individual 3D objects from the redesigned image using masks and depth.
    Saves each object as a PNG with transparency and returns spatial metadata.
    """
    objects_3d = []
    img_np = np.array(result_img)
    h, w = depth_map.shape

    # Scaling factor for depth (same as in ws.py)
    depth_float = depth_map.astype(np.float32) / 255.0
    depth_float = np.clip(depth_float, 0.01, 1.0)
    depth_metric = 0.3 + (1.0 - depth_float) * 4.7

    for i, m in enumerate(masks):
        mask_np = np.array(m["mask"])
        bbox = m["bbox"] # [x1, y1, x2, y2]
        label = m["label"]

        # 1. Extract object image with transparency
        # Create an RGBA image
        obj_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        obj_rgba[:, :, :3] = img_np
        obj_rgba[:, :, 3] = (mask_np * 255).astype(np.uint8)

        # Crop to bounding box
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop_rgba = obj_rgba[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop_rgba)
        
        # Save object texture
        obj_filename = f"obj_{client_id}_{i}_{label.replace(' ', '_')}.png"
        obj_path = os.path.join(output_dir, obj_filename)
        crop_pil.save(obj_path)

        # 2. Calculate 3D Position
        # Get center of mask
        coords = np.argwhere(mask_np)
        if len(coords) == 0:
            continue
            
        center_y, center_x = coords.mean(axis=0)
        obj_depth = depth_metric[int(center_y), int(center_x)]

        # Backproject to 3D
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        X = (center_x - cx) * obj_depth / fx
        Y = (center_y - cy) * obj_depth / fy
        Z = -obj_depth # Three.js uses negative Z for forward

        # 3. Calculate 3D Scale
        # Estimate width and height in 3D
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Approximate size in meters
        scale_x = (width_px * obj_depth) / fx
        scale_y = (height_px * obj_depth) / fy

        objects_3d.append({
            "id": i,
            "label": label,
            "texture_url": f"/outputs/{obj_filename}",
            "position": [float(X), float(-Y), float(Z)], # Flip Y for Three.js
            "scale": [float(scale_x), float(scale_y), 1.0],
            "is_structural": m.get("is_structural", False)
        })

    return objects_3d
