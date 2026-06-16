import numpy as np
import json
import os

def export_pointcloud_to_json(points, colors, output_path, max_points=80000):
    """
    Exports a point cloud to a JSON file optimized for web visualization.
    Downsamples the cloud if it exceeds max_points.
    """
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    
    # Flatten and convert to standard types for JSON serialization
    data = {
        "positions": points.flatten().tolist(),
        "colors": (colors.astype(np.float32) / 255.0).flatten().tolist(),
        "count": len(points)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    return os.path.basename(output_path)
