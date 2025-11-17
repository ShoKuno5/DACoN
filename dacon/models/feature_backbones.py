import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

try:
    from dacon.dift_integration import dift_features as dift_backend
except Exception:  # pragma: no cover - optional dependency
    try:
        dacon_root = Path(__file__).resolve().parents[2]
        if str(dacon_root) not in sys.path:
            sys.path.append(str(dacon_root))
        from dacon.dift_integration import dift_features as dift_backend
    except Exception:
        dift_backend = None


class FeatureBackend:
    """Base interface for feature extractors that feed DACoN."""

    def __init__(self, name: str, device: Optional[str] = None):
        self.name = name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def extract(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Extract feature maps for flattened (B*S, C, H, W) tensors."""
        raise NotImplementedError


class Dinov2FeatureBackend(FeatureBackend):
    """Wrapper around facebookresearch/dinov2 torch.hub checkpoints."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__("dinov2", device=cfg.get("device"))
        model_type = cfg.get("model_type", "dinov2_vitl14")
        self.input_size: Tuple[int, int] = tuple(cfg.get("input_size", (518, 518)))
        self.patch_size: int = int(cfg.get("patch_size", 14))
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # type: ignore
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = int(getattr(self.model, "embed_dim", cfg.get("embed_dim", 1024)))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @torch.no_grad()
    def extract(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        resized = F.interpolate(
            images, size=self.input_size, mode="bilinear", align_corners=False
        ).to(self.device)
        dino_output = self.model.get_intermediate_layers(
            resized, n=1, return_class_token=False
        )
        patch_tokens = dino_output[0]
        feat_h = self.input_size[0] // self.patch_size
        feat_w = self.input_size[1] // self.patch_size
        feats = patch_tokens.permute(0, 2, 1).contiguous().view(
            resized.shape[0], self.output_dim, feat_h, feat_w
        )
        return feats.to(target_device)


class GeoAwareFeatureBackend(FeatureBackend):
    """
    Uses the Stable-Diffusion + DINO fusion from GeoAware-SC to provide descriptors.
    Requires the GeoAware-SC repository to be present locally.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__("geoaware", device=cfg.get("device"))
        repo_path = Path(cfg.get("repo_path", "../GeoAware-SC")).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"GeoAware-SC repository not found at {repo_path}. "
                "Set network.feature_backbone.repo_path to a valid location."
            )
        if str(repo_path) not in sys.path:
            sys.path.append(str(repo_path))

        try:
            from model_utils.extractor_sd import load_model, process_features_and_mask
            from model_utils.extractor_dino import ViTExtractor
            from model_utils.projection_network import AggregationNetwork
            from utils.utils_correspondence import resize as ga_resize
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Failed to import GeoAware-SC modules. "
                "Please ensure its Python dependencies are installed."
            ) from exc

        self.resize_fn = ga_resize
        self.process_features_and_mask = process_features_and_mask

        self.num_patches: int = int(cfg.get("num_patches", 60))
        self.sd_image_size: int = int(cfg.get("sd_image_size", self.num_patches * 16))
        self.sd_num_timesteps: int = int(cfg.get("sd_num_timesteps", 50))
        self.sd_diffusion_ver: str = cfg.get("sd_diffusion_ver", "v1-5")
        self.dino_stride: int = int(cfg.get("geo_dino_stride", 14))
        self.dino_model_type: str = cfg.get("geo_dino_model", "dinov2_vitb14")
        self.dino_layer: int = int(cfg.get("geo_dino_layer", 11))

        agg_feature_dims: Sequence[int] = cfg.get(
            "aggregation_feature_dims", [640, 1280, 1280, 768]
        )
        agg_projection_dim: int = int(cfg.get("aggregation_output_dim", 768))
        pretrained_weights: Optional[str] = cfg.get("pretrained_weights")
        if pretrained_weights is None:
            raise ValueError(
                "GeoAware backend requires 'pretrained_weights' "
                "(path to AggregationNetwork checkpoint)."
            )
        weight_path = Path(pretrained_weights).expanduser().resolve()
        if not weight_path.exists():
            raise FileNotFoundError(
                f"GeoAware pretrained weights not found: {weight_path}"
            )

        self.sd_model, self.sd_aug = load_model(
            diffusion_ver=self.sd_diffusion_ver,
            image_size=self.sd_image_size,
            num_timesteps=self.sd_num_timesteps,
        )
        self.sd_model.to(self.device)
        self.sd_model.eval()

        self.vit_extractor = ViTExtractor(
            self.dino_model_type, stride=self.dino_stride, device=str(self.device)
        )
        self.aggre_net = AggregationNetwork(
            feature_dims=list(agg_feature_dims),
            projection_dim=agg_projection_dim,
            device=str(self.device),
        )
        state_dict = torch.load(weight_path, map_location=self.device)
        self.aggre_net.load_pretrained_weights(state_dict)
        self.aggre_net.eval()
        self._output_dim = agg_projection_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        img = image.detach().cpu().clamp(0.0, 1.0)
        img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img)

    @torch.no_grad()
    def _encode_single(self, tensor_img: torch.Tensor) -> torch.Tensor:
        pil_img = self._tensor_to_pil(tensor_img)

        img_sd_input = self.resize_fn(
            pil_img, target_res=self.sd_image_size, resize=True, to_pil=True
        )
        features_sd = self.process_features_and_mask(
            self.sd_model, self.sd_aug, img_sd_input, mask=False, raw=True
        )
        features_sd = {k: v.to(self.device) for k, v in features_sd.items()}
        features_sd.pop("s2", None)

        img_dino_input = self.resize_fn(
            pil_img,
            target_res=self.num_patches * self.dino_stride,
            resize=True,
            to_pil=True,
        )
        img_batch = self.vit_extractor.preprocess_pil(img_dino_input).to(self.device)
        features_dino = self.vit_extractor.extract_descriptors(
            img_batch, layer=self.dino_layer, facet="token"
        )
        features_dino = (
            features_dino.to(self.device)
            .permute(0, 1, 3, 2)
            .reshape(1, -1, self.num_patches, self.num_patches)
        )

        desc_gathered = torch.cat(
            [
                features_sd["s3"],
                F.interpolate(
                    features_sd["s4"],
                    size=(self.num_patches, self.num_patches),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    features_sd["s5"],
                    size=(self.num_patches, self.num_patches),
                    mode="bilinear",
                    align_corners=False,
                ),
                features_dino,
            ],
            dim=1,
        )
        desc = self.aggre_net(desc_gathered.to(self.device))
        desc = desc / (torch.linalg.norm(desc, dim=1, keepdim=True) + 1e-8)
        return desc

    def extract(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        outputs = []
        for img in images:
            outputs.append(self._encode_single(img))
        feat_map = torch.cat(outputs, dim=0)
        return feat_map.to(target_device)


class SdDinoFeatureBackend(FeatureBackend):
    """
    Uses the Stable-Diffusion descriptors from the sd-dino project.
    Designed for feature-backbone-only mode (no UNet fusion).
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__("sd-dino", device=cfg.get("device"))
        repo_path = Path(cfg.get("repo_path", "../sd-dino")).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"sd-dino repository not found at {repo_path}. "
                "Set network.feature_backbone.repo_path to a valid location."
            )
        if str(repo_path) not in sys.path:
            sys.path.append(str(repo_path))
        mask2former_path = repo_path / "third_party" / "Mask2Former"
        if mask2former_path.exists() and str(mask2former_path) not in sys.path:
            sys.path.append(str(mask2former_path))

        try:
            from extractor_sd import load_model, process_features_and_mask
            from utils.utils_correspondence import resize as sd_resize
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Failed to import sd-dino modules. "
                "Please ensure its dependencies (Detectron2/Mask2Former) are installed."
            ) from exc

        self.resize_fn = sd_resize
        self.process_features_and_mask = process_features_and_mask

        self.sd_diffusion_ver: str = cfg.get("sd_diffusion_ver", "v1-5")
        self.sd_image_size: int = int(cfg.get("sd_image_size", 960))
        self.sd_num_timesteps: int = int(cfg.get("sd_num_timesteps", 50))
        self.sd_block_indices: Sequence[int] = tuple(
            cfg.get("sd_block_indices", (2, 5, 8, 11))
        )
        self.sd_decoder_only: bool = bool(cfg.get("sd_decoder_only", True))
        self.sd_encoder_only: bool = bool(cfg.get("sd_encoder_only", False))
        self.sd_resblock_only: bool = bool(cfg.get("sd_resblock_only", False))
        self.combine_s4_s5: bool = bool(cfg.get("combine_s4_s5", True))
        self.normalize_output: bool = bool(cfg.get("normalize", True))

        self.model, self.aug = load_model(
            diffusion_ver=self.sd_diffusion_ver,
            image_size=self.sd_image_size,
            num_timesteps=self.sd_num_timesteps,
            block_indices=self.sd_block_indices,
            decoder_only=self.sd_decoder_only,
            encoder_only=self.sd_encoder_only,
            resblock_only=self.sd_resblock_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self._output_dim = cfg.get("output_dim")

    @property
    def output_dim(self) -> Optional[int]:
        return self._output_dim

    @staticmethod
    def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
        img = image.detach().cpu().clamp(0.0, 1.0)
        img = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(img)

    @torch.no_grad()
    def _encode_single(self, tensor_img: torch.Tensor) -> torch.Tensor:
        pil_img = self._tensor_to_pil(tensor_img)
        img_input = self.resize_fn(
            pil_img, target_res=self.sd_image_size, resize=True, to_pil=True
        )
        features = self.process_features_and_mask(
            self.model, self.aug, img_input, mask=False, raw=True
        )
        desc = features["s4"]
        if self.combine_s4_s5 and "s5" in features:
            desc = torch.cat(
                [
                    desc,
                    F.interpolate(
                        features["s5"],
                        size=desc.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                dim=1,
            )
        if self.normalize_output:
            desc = desc / (torch.linalg.norm(desc, dim=1, keepdim=True) + 1e-8)
        return desc

    def extract(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        outputs = []
        for img in images:
            outputs.append(self._encode_single(img))
        feat_map = torch.cat(outputs, dim=0)
        return feat_map.to(target_device)


class DiftFeatureBackend(FeatureBackend):
    """
    Reuses the baseline DIFT extractor as a FeatureBackend so it can participate
    in feature_backbone_only evaluations.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__("dift", device=cfg.get("device"))
        if dift_backend is None:
            raise ImportError(
                "DIFT backend is not available. Clone DACoN/dift or set DACON_DIFT_PATH/FGW_DIFT_PATH."
            )

        self.img_size: Tuple[int, int] = tuple(cfg.get("img_size", (768, 768)))
        self.model_id: str = cfg.get("model_id", "stabilityai/stable-diffusion-2-1")
        self.t: int = int(cfg.get("t", 261))
        self.up_ft_index: int = int(cfg.get("up_ft_index", 1))
        self.ensemble_size: int = int(cfg.get("ensemble_size", 8))
        self.prompt: str = cfg.get("prompt", "")
        self.cache_dir: Optional[str] = cfg.get("cache_dir")
        self.output_channels: int = int(cfg.get("output_dim", 640))

    @property
    def output_dim(self) -> int:
        return self.output_channels

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        return to_pil_image(image.detach().cpu().clamp(0.0, 1.0))

    def _encode_single(self, tensor_img: torch.Tensor) -> torch.Tensor:
        pil_img = self._to_pil(tensor_img)
        feature_map = dift_backend.extract_dift_features(
            img=pil_img,
            img_size=self.img_size,
            t=self.t,
            up_ft_index=self.up_ft_index,
            ensemble_size=self.ensemble_size,
            prompt=self.prompt,
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            device=str(self.device),
        )
        if not isinstance(feature_map, torch.Tensor):
            feature_map = torch.from_numpy(feature_map)
        return feature_map.to(self.device).float()

    def extract(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        outputs = []
        for img in images:
            outputs.append(self._encode_single(img))
        feat_map = torch.stack(outputs, dim=0)
        return feat_map.to(target_device)


def build_feature_backend(network_cfg: Dict[str, Any]) -> FeatureBackend:
    """Factory that keeps backward compatibility with legacy configs."""
    backbone_cfg = network_cfg.get("feature_backbone")
    legacy_type = "dinov2"
    if backbone_cfg is None:
        backbone_cfg = {
            "type": legacy_type,
            "model_type": network_cfg.get("dino_model_type", "dinov2_vitl14"),
            "input_size": network_cfg.get("dino_input_size", (518, 518)),
        }
    backbone_type = backbone_cfg.get("type", legacy_type).lower()

    if backbone_type == "geoaware":
        return GeoAwareFeatureBackend(backbone_cfg)
    if backbone_type == "sd-dino":
        return SdDinoFeatureBackend(backbone_cfg)
    if backbone_type == "dift":
        return DiftFeatureBackend(backbone_cfg)
    if backbone_type == "dinov2":
        backbone_cfg.setdefault(
            "model_type", network_cfg.get("dino_model_type", "dinov2_vitl14")
        )
        backbone_cfg.setdefault(
            "input_size", network_cfg.get("dino_input_size", (518, 518))
        )
        return Dinov2FeatureBackend(backbone_cfg)

    raise ValueError(f"Unsupported feature_backbone type: {backbone_type}")
