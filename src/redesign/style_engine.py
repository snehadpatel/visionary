import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial.distance import braycurtis

class StyleEngine:
    """
    Classical CV Style Engine.
    Matches detected clusters to library furniture via HSV Histograms + Dominant Colors.
    """
    def __init__(self, index_path: str, signatures_path: str = None):
        self.project_root = Path("/Users/snehapatel/visionary")
        with open(index_path, 'r') as f:
            self.library = json.load(f)
            
        self.signatures = {}
        if signatures_path and os.path.exists(signatures_path):
            with open(signatures_path, 'r') as f:
                self.signatures = json.load(f)
            
    def get_style_inspiration(self, style):
        """Returns palette and starter pack for a style."""
        return self.signatures.get(style, {
            "palette": ["#FFFFFF", "#CCCCCC", "#888888"],
            "starter_pack": ["sofa", "table", "chair"]
        })
            
    def _compare_histograms(self, hist1, hist2):
        """
        Bray-Curtis similarity for histograms (higher is better).
        hist: (1024,) normalized
        """
        dist = braycurtis(hist1, hist2)
        return 1.0 - dist

    def find_matches(self, cluster, target_style=None, top_k=3):
        """
        cluster: FurnitureCluster object from dbscan.py
        """
        matches = []
        
        for item in self.library:
            if item['type'] != cluster.ftype and cluster.ftype != 'unknown':
                continue
                
            # 1. Style Score
            style_score = 0.0
            if target_style:
                if item['style'] == target_style:
                    style_score = 1.0
                elif item['style'] == 'transitional':
                    style_score = 0.5
            
            # 2. Color Score (compare HSV histograms)
            hist_path = self.project_root / item['color_hist_path']
            if hist_path.exists():
                lib_hist = np.load(hist_path)
                color_score = self._compare_histograms(cluster.color_hist, lib_hist)
            else:
                color_score = 0.0
                
            # 3. Volume Score (similarity in size)
            lib_bbox_min = np.array(item['bbox']['min'])
            lib_bbox_max = np.array(item['bbox']['max'])
            lib_volume = np.prod(lib_bbox_max - lib_bbox_min)
            vol_score = 1.0 - min(1.0, abs(cluster.volume - lib_volume) / (cluster.volume + 1e-8))
            
            final_score = (0.4 * color_score + 
                           0.4 * style_score + 
                           0.2 * vol_score)
            
            matches.append({
                "item": item,
                "score": float(final_score),
                "color_sim": float(color_score),
                "vol_sim": float(vol_score)
            })
            
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k]

class RedesignEngine:
    """
    Orchestrates the furniture replacement and style transfer.
    """
    def __init__(self, style_engine: StyleEngine):
        self.style_engine = style_engine

    def redesign_scene(self, scene_data, target_style):
        """
        scene_data: dict containing 'floor', 'walls', 'clusters', 'room_type'
        """
        redesigned_furniture = []
        inspiration = self.style_engine.get_style_inspiration(target_style)
        
        # A. Existing Furniture Replacement
        if scene_data.get('clusters'):
            for cluster in scene_data['clusters']:
                matches = self.style_engine.find_matches(cluster, target_style)
                if not matches: continue
                
                best_match = matches[0]
                redesigned_furniture.append({
                    "original_label": cluster.label,
                    "original_type": cluster.ftype,
                    "replacement_id": best_match['item']['id'],
                    "replacement_name": best_match['item']['product_name'],
                    "replacement_url": best_match['item']['product_url'],
                    "centroid": list(cluster.centroid),
                    "score": best_match['score']
                })
        
        # B. Empty Room Support: Suggest furniture if none found
        else:
            room_type = scene_data.get('room_type', 'living_room')
            starter_types = inspiration.get('starter_pack', [])
            
            # Find one best item for each starter type in the target style
            for ftype in starter_types:
                # Simple lookup in library for best style match
                items = [it for it in self.style_engine.library if it['type'] == ftype and it['style'] == target_style]
                if not items: # Fallback to any style
                    items = [it for it in self.style_engine.library if it['type'] == ftype]
                
                if items:
                    best_item = items[0] # Just pick first for suggestion
                    redesigned_furniture.append({
                        "original_label": -1,
                        "original_type": "suggestion",
                        "replacement_id": best_item['id'],
                        "replacement_name": best_item['product_name'],
                        "replacement_url": best_item['product_url'],
                        "centroid": [0, 0, 0], # Placeholder for new item
                        "score": 1.0
                    })

        return {
            "style": target_style,
            "replacements": redesigned_furniture,
            "recommended_palette": inspiration.get('palette', []),
            "floor_color": inspiration.get('palette', ["#D2B48C"])[0],
            "wall_color": inspiration.get('palette', ["#F5F5F5"])[-1]
        }
