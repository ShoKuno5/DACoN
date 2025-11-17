"""DIFT feature extraction and vertex sampling (standalone integration)."""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_GLOBAL_SD_MODEL = None
_GLOBAL_SD_MODEL_ID = None


def _resolve_dift_paths() -> List[Path]:
    """Build a list of candidate directories that may contain the DIFT repo."""
    candidates: List[Path] = []
    env_override = os.getenv("FGW_DIFT_PATH") or os.getenv("DACON_DIFT_PATH")
    if env_override:
        candidates.append(Path(env_override))

    # DACoN root -> DACoN/dift (user clone) or DACoN/third_party/dift (optional)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "dift")
    candidates.append(repo_root / "third_party" / "dift")

    # Filter duplicates while preserving order
    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)
    return unique_candidates


def get_or_create_sd_model(model_id: str = "stabilityai/stable-diffusion-2-1", device: Optional[str] = None):
    """Get or create a singleton SDFeaturizer model."""
    global _GLOBAL_SD_MODEL, _GLOBAL_SD_MODEL_ID

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.getenv("DIFT_MOCK", "0") == "1":
        print("Warning: DIFT_MOCK=1, using mock DIFT features")
        _GLOBAL_SD_MODEL = None
        _GLOBAL_SD_MODEL_ID = model_id
        return _GLOBAL_SD_MODEL

    if _GLOBAL_SD_MODEL is None or _GLOBAL_SD_MODEL_ID != model_id:
        try:
            # Preferred import when 'dift' is installed as a package
            from dift.src.models.dift_sd import SDFeaturizer
        except Exception:
            try:
                import sys

                for candidate in _resolve_dift_paths():
                    if candidate.is_dir() and str(candidate) not in sys.path:
                        sys.path.append(str(candidate))
                from src.models.dift_sd import SDFeaturizer  # type: ignore
            except Exception as exc:
                print(f"Warning: DIFT not available, using mock model (import error: {exc!r})")
                _GLOBAL_SD_MODEL = None
                _GLOBAL_SD_MODEL_ID = model_id
            else:
                _GLOBAL_SD_MODEL = SDFeaturizer(model_id)
                try:
                    if device == "cuda" and hasattr(_GLOBAL_SD_MODEL, "to"):
                        _GLOBAL_SD_MODEL = _GLOBAL_SD_MODEL.to(device)
                except Exception:
                    pass
                _GLOBAL_SD_MODEL_ID = model_id
        else:
            _GLOBAL_SD_MODEL = SDFeaturizer(model_id)
            try:
                if device == "cuda" and hasattr(_GLOBAL_SD_MODEL, "to"):
                    _GLOBAL_SD_MODEL = _GLOBAL_SD_MODEL.to(device)
            except Exception:
                pass
            _GLOBAL_SD_MODEL_ID = model_id

    return _GLOBAL_SD_MODEL


def compute_image_hash(img: Image.Image) -> str:
    img_bytes = img.tobytes()
    hash_obj = hashlib.md5(img_bytes)
    return hash_obj.hexdigest()[:8]


def extract_dift_features(
    img: Image.Image,
    img_size: Tuple[int, int] = (768, 768),
    t: int = 261,
    up_ft_index: int = 1,
    ensemble_size: int = 8,
    prompt: str = "",
    model_id: str = "stabilityai/stable-diffusion-2-1",
    cache_dir: Optional[str] = None,
    img_identifier: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_path = None
    if cache_dir is not None:
        if img_identifier is None:
            img_identifier = compute_image_hash(img)
        cache_filename = f"dift_{img_identifier}_t{t}_up{up_ft_index}_ens{ensemble_size}.pt"
        cache_path = Path(cache_dir) / cache_filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return torch.load(cache_path, map_location=device)

    dift = get_or_create_sd_model(model_id, device)
    if dift is None:
        return torch.randn(640, img_size[0] // 8, img_size[1] // 8, device=device)

    img_resized = img.resize(img_size, Image.LANCZOS)
    from torchvision.transforms import PILToTensor

    img_tensor = PILToTensor()(img_resized).float() / 255.0
    img_tensor = (img_tensor - 0.5) * 2.0
    img_tensor = img_tensor.unsqueeze(0)
    if device == "cuda":
        img_tensor = img_tensor.to(device)

    with torch.no_grad():
        features = dift.forward(
            img_tensor,
            prompt=prompt,
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    features = features.squeeze(0)
    if cache_path is not None:
        torch.save(features.cpu(), cache_path)

    return features


def load_cached_features(cache_path: str) -> torch.Tensor:
    return torch.load(cache_path, map_location="cpu")


def sample_features_at_vertices(
    feature_map: torch.Tensor,
    vertices: np.ndarray,
    img_size: Tuple[int, int],
) -> np.ndarray:
    device = feature_map.device
    C, H_feat, W_feat = feature_map.shape
    N = len(vertices)

    vertices_norm = vertices.copy()
    vertices_norm[:, 0] = 2.0 * vertices[:, 0] / (img_size[1] - 1) - 1.0
    vertices_norm[:, 1] = 2.0 * vertices[:, 1] / (img_size[0] - 1) - 1.0
    vertices_norm = np.clip(vertices_norm, -1.0, 1.0)

    grid = torch.from_numpy(vertices_norm).float().to(device)
    grid = grid.view(1, 1, N, 2)

    feature_map_batch = feature_map.unsqueeze(0)
    sampled = F.grid_sample(
        feature_map_batch,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)
    return sampled.detach().cpu().numpy()


def extract_vertex_features(
    img: Image.Image,
    vertices: np.ndarray,
    img_size: Tuple[int, int] = (768, 768),
    t: int = 261,
    up_ft_index: int = 1,
    ensemble_size: int = 8,
    prompt: str = "",
    feat_cache_dir: Optional[str] = None,
    feat_cache_path: Optional[str] = None,
    img_identifier: Optional[str] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    if feat_cache_path and os.path.exists(feat_cache_path):
        feature_map = load_cached_features(feat_cache_path)
    else:
        feature_map = extract_dift_features(
            img=img,
            img_size=img_size,
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
            prompt=prompt,
            cache_dir=feat_cache_dir,
            img_identifier=img_identifier,
            device=device,
        )

    orig_img_size = (img.size[1], img.size[0])

    if isinstance(feature_map, torch.Tensor):
        feature_map = F.interpolate(
            feature_map.unsqueeze(0), size=orig_img_size, mode="bilinear", align_corners=True
        ).squeeze(0)

    vertex_features = sample_features_at_vertices(
        feature_map=feature_map,
        vertices=vertices,
        img_size=orig_img_size,
    )
    return vertex_features


def normalize_features(features: np.ndarray, method: str = "l2") -> np.ndarray:
    if method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return features / norms
    if method == "standard":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        return (features - mean) / std
    return features

