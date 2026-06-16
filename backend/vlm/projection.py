"""
Visionary VLM — Projection MLP.
Projects CLIP image embeddings (768-dim) into TinyLlama's language space (2048-dim).
This is the layer trained FROM SCRATCH on interior design data.

Architecture: 768 → 1536 → 2048 with GELU activations and LayerNorm
"""
import torch
import torch.nn as nn
from pathlib import Path


class ProjectionMLP(nn.Module):
    """
    Projects CLIP image embeddings into TinyLlama's language embedding space.
    
    This 2-layer MLP with GELU + LayerNorm is the trainable bridge between
    the frozen vision encoder and frozen language decoder. It learns to map
    visual concepts (room layout, furniture style, lighting) into the
    semantic space that TinyLlama understands.
    """
    def __init__(self, clip_dim: int = 768, llm_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim, 1536),
            nn.GELU(),
            nn.LayerNorm(1536),
            nn.Linear(1536, llm_dim),
            nn.GELU(),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CLIP image embedding of shape (B, 768)
        Returns:
            Projected embedding of shape (B, 2048) in TinyLlama's space
        """
        out = self.net(x)
        # Ensure output matches the decoder's expected dtype (e.g. float16)
        if hasattr(self, "_target_dtype"):
            return out.to(self._target_dtype)
        return out


def load_projection(weights_path: str = None) -> ProjectionMLP:
    """
    Load the projection MLP, with graceful fallback to random init.
    
    Args:
        weights_path: Path to trained projection weights (.pt file)
    
    Returns:
        ProjectionMLP module (eval mode)
    """
    if weights_path is None:
        # Default path relative to project root
        weights_path = str(Path(__file__).resolve().parent.parent.parent / "models" / "projection_layer.pt")
    
    proj = ProjectionMLP()
    
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        proj.load_state_dict(state_dict)
        print(f"[VLM Projection] Loaded trained projection weights from {weights_path}")
    except FileNotFoundError:
        print(f"[VLM Projection] No projection weights found at {weights_path}")
        print("[VLM Projection] Using random initialization — run training/train_projection.py first for best results.")
    except Exception as e:
        print(f"[VLM Projection] Error loading weights: {e}. Using random init.")
    
    proj.eval()
    return proj
