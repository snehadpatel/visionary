import cv2
import torch
import numpy as np
import os
from pathlib import Path

from src.models.fast_depth_net import FastDepthNet
from src.models.unet import UNet
from src.geometry.backprojector import estimate_intrinsics, depth_to_pointcloud_np, clean_pointcloud_np
from src.geometry.ransac import detect_room_planes
from src.geometry.dbscan import dbscan_3d, extract_clusters
from src.redesign.custom_generator import CustomNeuralGenerator
from src.redesign.style_engine import StyleEngine, RedesignEngine

class VisionaryPipeline:
    def __init__(
        self,
        depth_weights="models/depth_model.pth",
        seg_weights="models/seg_model.pth",
        redesign_weights="models/redesign_generator.pth",
        furniture_index="data/annotations/furniture_index.json",
        style_signatures="data/annotations/style_signatures.json",
        visualize=False,
        use_sd=False,
        device=None
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Visionary Pipeline initializing on {self.device}...")
        
        project_root = Path("/Users/snehapatel/visionary")
        self.use_sd = use_sd
        
        # 1. Models
        self.depth_net = FastDepthNet().to(self.device)
        if os.path.exists(project_root / depth_weights):
            self.depth_net.load_state_dict(torch.load(project_root / depth_weights, map_location=self.device))
        self.depth_net.eval()
        
        self.seg_net = UNet(n_classes=21).to(self.device)
        if os.path.exists(project_root / seg_weights):
            self.seg_net.load_state_dict(torch.load(project_root / seg_weights, map_location=self.device))
        self.seg_net.eval()
        
        self.redesign_weights = str(project_root / redesign_weights)
        
        # 2. Engines
        signatures_path = project_root / style_signatures if style_signatures else None
        self.style_engine = StyleEngine(str(project_root / furniture_index), str(signatures_path) if signatures_path else None)
        self.redesign_engine = RedesignEngine(self.style_engine)
        from src.redesign.furniture_placer import FurniturePlacer
        self.furniture_placer = FurniturePlacer(self.style_engine.library)
        
        # 3. Generator
        self.image_generator = None
        self.sd_generator = None
        
        if visualize:
            self.image_generator = CustomNeuralGenerator(self.redesign_weights, device=self.device)
            if self.use_sd:
                # Lazy load SD generator only if needed
                from backend.pipeline.image_generator import generate_redesigned_image
                self.sd_generator = generate_redesigned_image

    def process_room(self, image_path: str, target_style: str = "scandinavian", user_prompt: str = "", generate_image: bool = False, use_sd: bool = False):
        """
        Full Pipeline: Image -> Redesign Result
        """
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Use provided or default
        active_use_sd = use_sd or self.use_sd
            
        # A. Neural Inference
        print("Step 1: Depth & Segmentation Inference...")
        depth_metric = self.depth_net.infer(image_bgr, device=self.device)
        seg_mask = self.seg_net.infer(image_bgr, device=self.device)
        
        # B. Geometry Processing
        print("Step 2: 3D Back-projection & Plane Detection...")
        h, w = image_bgr.shape[:2]
        K = estimate_intrinsics(w, h)
        points, colors = depth_to_pointcloud_np(image_bgr, depth_metric, K)
        points, colors = clean_pointcloud_np(points, colors)
        room_data = detect_room_planes(points)
        
        # C. Furniture Clustering
        print("Step 3: Furniture Extraction & Clustering...")
        # Map (u, v) back to seg_mask
        u_all = (points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]).astype(int).clip(0, w-1)
        v_all = (points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]).astype(int).clip(0, h-1)
        seg_labels_flat = seg_mask[v_all, u_all]
        
        furniture_points = points[room_data['furniture_mask']]
        furniture_colors = colors[room_data['furniture_mask']]
        furniture_seg = seg_labels_flat[room_data['furniture_mask']]
        
        cluster_labels = dbscan_3d(furniture_points)
        clusters = extract_clusters(
            furniture_points, furniture_colors, cluster_labels, 
            furniture_seg, room_data['floor_height']
        )
        
        # Populate masks for clusters
        for c in clusters:
            mask_2d = np.zeros((h, w), dtype=np.uint8)
            c_pts = c.points
            c_u, c_v = self._project_points(c_pts, K, h, w)
            mask_2d[c_v, c_u] = 255
            c.mask = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1) > 0

        # D. Style Engine
        print(f"Step 4: Style Analysis ({target_style})...")
        scene_data = {'floor': room_data['floor_plane'], 'walls': room_data['wall_planes'], 'clusters': clusters}
        redesign_result = self.redesign_engine.redesign_scene(scene_data, target_style)
        
        # E. 3D Reconfiguration & Synthesis
        print("Step 5: 3D Scene Reconfiguration...")
        assets_to_render = []
        
        # Determine layout Strategy
        if "layout" in user_prompt.lower() or "auto" in user_prompt.lower():
            suggested_layout = self.furniture_placer.suggest_layout_heuristic(room_data, target_style)
            for suggestion in suggested_layout:
                # Find matching library item for this type
                item = next((it for it in self.style_engine.library if it['type'] == suggestion['type'] and it['style'] == target_style), None)
                if not item: continue
                
                pts, _ = self.furniture_placer.load_ply_binary(str(Path("/Users/snehapatel/visionary") / item['ply_path']))
                if pts is not None:
                    T = self.furniture_placer.get_transform_matrix(suggestion['pos'], suggestion['rot'])
                    assets_to_render.append({'points': pts, 'transform': T})
        else:
            # Traditional placement: swap existing clusters with library counterparts
            for repl in redesign_result['replacements']:
                item = next((it for it in self.style_engine.library if it['id'] == repl['replacement_id']), None)
                if not item: continue
                
                pts, _ = self.furniture_placer.load_ply_binary(str(Path("/Users/snehapatel/visionary") / item['ply_path']))
                if pts is not None:
                    T = self.furniture_placer.get_transform_matrix(repl['centroid'], 0) # Maintain original orientation for now
                    assets_to_render.append({'points': pts, 'transform': T})
        
        synthetic_depth, synthetic_mask = self.furniture_placer.synthesize_scene(assets_to_render, K, h, w)
        
        # F. Custom Neural Image Generation
        visualized_path = None
        if generate_image:
            print("Step 6: Executing Custom Neural Redesign with Synthetic Priors...")
            if self.image_generator is None:
                self.image_generator = CustomNeuralGenerator(self.redesign_weights, device=self.device)
            
            from PIL import Image
            orig_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            
            # Combine the synthetic mask with the original furniture mask for "Void Inpainting"
            # This tells the model to ignore the OLD furniture and synthesis the NEW furniture
            furniture_labels = [7, 9, 11, 15, 18, 20]
            orig_furniture_mask = np.isin(seg_mask, furniture_labels).astype(np.uint8) * 255
            combined_mask = np.maximum(orig_furniture_mask, synthetic_mask)
            
            # 2. Run Generator
            if active_use_sd:
                print("🎨 Running High-Quality Stable Diffusion Engine...")
                if self.sd_generator is None:
                    from backend.pipeline.image_generator import generate_redesigned_image
                    self.sd_generator = generate_redesigned_image
                
                from backend.pipeline.sd_prompt_builder import build_sd_prompt
                from backend.pipeline.budget_engine import allocate_budget
                # Minimal mock budget/vlm for direct pipeline calls
                budget_plan = allocate_budget(50000, {}, target_style, user_prompt)
                sd_prompt, negative_prompt = build_sd_prompt(target_style, budget_plan, {}, user_prompt)
                
                result_pil = self.sd_generator(
                    orig_pil,
                    synthetic_depth,
                    [combined_mask > 0],
                    sd_prompt,
                    negative_prompt
                )
            else:
                print("⚡ Running High-Speed Custom Neural Engine (30 FPS)...")
                if self.image_generator is None:
                    self.image_generator = CustomNeuralGenerator(self.redesign_weights, device=self.device)
                
                result_pil = self.image_generator.generate_redesign(
                    orig_pil, 
                    synthetic_depth, 
                    [combined_mask > 0]
                )
            
            out_dir = Path(image_path).parent / "redesigned"
            out_dir.mkdir(exist_ok=True)
            visualized_path = str(out_dir / f"{Path(image_path).stem}_{'sd' if active_use_sd else 'custom'}_redesign.jpg")
            result_pil.save(visualized_path)
            redesign_result['visualized_image'] = visualized_path

        return {
            "depth_map": depth_metric,
            "synthetic_depth": synthetic_depth,
            "seg_mask": seg_mask,
            "redesign": redesign_result,
            "point_cloud": {"points": points, "colors": colors}
        }


    def _project_points(self, points: np.ndarray, intrinsics: np.ndarray, h: int, w: int):
        import numpy as np
        K = intrinsics
        # pts_homo = np.hstack([points, np.ones((len(points), 1))]) # Not needed if multiplying K @ points.T
        p_img = (K @ points.T).T
        u = (p_img[:, 0] / p_img[:, 2]).astype(int)
        v = (p_img[:, 1] / p_img[:, 2]).astype(int)
        
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u[valid], v[valid]

def scene_data_for_spec(clusters, style):
    """Bridge between geometric clusters and spec engine expectations."""
    objects = []
    for c in clusters:
        objects.append({
            "label": c.ftype,
            "confidence": 0.9, # Mocked for now
        })
    return {
        "current_style": style,
        "objects": objects
    }

if __name__ == "__main__":
    # Test on a single image
    pipeline = VisionaryPipeline()
    # test_img = "/Users/snehapatel/visionary/data/datasets/raw/small_bedroom_design/image_0.jpg"
    # result = pipeline.process_room(test_img, "bohemian")
    # print(f"Redesign complete with {len(result['redesign']['replacements'])} replacements.")
