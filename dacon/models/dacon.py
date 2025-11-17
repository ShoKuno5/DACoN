import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from utils import segment_pooling
from .feature_backbones import build_feature_backend

try:
    from dacon.dift_integration import dift_features as dift_backend
except Exception:  # pragma: no cover - optional dependency
    try:
        dacon_root = Path(__file__).resolve().parents[1]
        if str(dacon_root) not in sys.path:
            sys.path.append(str(dacon_root))
        from dacon.dift_integration import dift_features as dift_backend
    except Exception:
        dift_backend = None

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
    
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), 
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_list=[64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_list = hidden_dim_list
        self.num_down_blocks = len(hidden_dim_list)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.encoder_blocks = nn.ModuleList()
        current_input_dim = input_dim
        for current_output_dim in hidden_dim_list:
            self.encoder_blocks.append(ConvBlock(current_input_dim, current_output_dim))
            current_input_dim = current_output_dim

        bottleneck_input_dim = hidden_dim_list[-1]
        bottleneck_output_dim = bottleneck_input_dim * 2
        self.bottleneck = ConvBlock(bottleneck_input_dim, bottleneck_output_dim)
        
        self.decoder_up_convs = nn.ModuleList()
        self.decoder_conv_blocks = nn.ModuleList()

        current_decoder_input_dim = bottleneck_output_dim 
        
        for i in reversed(range(self.num_down_blocks)):
            skip_connection_dim = hidden_dim_list[i]
            upconv_output_dim = skip_connection_dim 
            self.decoder_up_convs.append(UpConv(current_decoder_input_dim, upconv_output_dim))
            self.decoder_conv_blocks.append(ConvBlock(upconv_output_dim * 2, upconv_output_dim))
            current_decoder_input_dim = upconv_output_dim 
            
        self.final_conv = nn.Conv2d(current_decoder_input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x) 
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections_reversed = skip_connections[::-1]
        
        for i in range(self.num_down_blocks):
            skip_connection = skip_connections_reversed[i] 
            
            x = self.decoder_up_convs[i](x)
            x = torch.cat([skip_connection, x], dim=1) 
            x = self.decoder_conv_blocks[i](x)

        x = self.final_conv(x)
        return x


class DACoNModel(nn.Module):
    def __init__(self, dacon_config, version):
        super(DACoNModel, self).__init__()

        self.version = version
        self.feature_backend = build_feature_backend(dacon_config)
        self.feature_backbone_only = bool(dacon_config.get("feature_backbone_only", False))
        self.dino_dim = getattr(self.feature_backend, "output_dim", None)
        self.feats_dim = dacon_config["feats_dim"]

        self.unet_input_size = tuple(dacon_config["unet_input_size"])
        self.segment_pool_size = tuple(dacon_config["segment_pool_size"])
        self.unet_hidden_dim_list = dacon_config["unet_hidden_dim_list"]

        if not self.feature_backbone_only:
            if self.dino_dim is None:
                raise ValueError(
                    "The selected feature backbone must define 'output_dim' unless "
                    "'feature_backbone_only' is enabled."
                )
            self.unet = UNet(
                input_dim=3, output_dim=self.feats_dim, hidden_dim_list=self.unet_hidden_dim_list
            )
            self.dino_mlp = MLP(self.dino_dim, self.feats_dim * 4, self.feats_dim)
            if self.version == "1_0":
                self.dino_unet_mlp = MLP(self.feats_dim * 2, self.feats_dim * 4, self.feats_dim)
            else:
                self.dino_unet_mlp = None
        else:
            self.unet = None
            self.dino_mlp = None
            self.dino_unet_mlp = None

        dift_cfg = dacon_config.get("dift", {})
        self.use_dift = bool(dift_cfg.get("enabled", False))
        raw_dift_img_size = dift_cfg.get("img_size", (768, 768))
        if raw_dift_img_size is None:
            raw_dift_img_size = (768, 768)
        self.dift_img_size: Tuple[int, int] = tuple(raw_dift_img_size)
        self.dift_model_id: str = dift_cfg.get("model_id", "stabilityai/stable-diffusion-2-1")
        self.dift_t: int = int(dift_cfg.get("t", 261))
        self.dift_up_ft_index: int = int(dift_cfg.get("up_ft_index", 1))
        self.dift_ensemble_size: int = int(dift_cfg.get("ensemble_size", 8))
        self.dift_prompt: str = dift_cfg.get("prompt", "")
        self.dift_cache_dir: Optional[str] = dift_cfg.get("cache_dir")
        self.dift_device: Optional[str] = dift_cfg.get("device")

        self.latest_dift_feats_map: Optional[torch.Tensor] = None
        self.latest_seg_dift_feats: Optional[torch.Tensor] = None
        self.latest_seg_dift_feats_src: Optional[torch.Tensor] = None
        self.latest_seg_dift_feats_tgt: Optional[torch.Tensor] = None
        self.latest_dift_sim_map: Optional[torch.Tensor] = None

    def l2_normalize(self, x, dim=-1, eps=1e-6):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    
    def _prepare_images(self, images, target_size):
        images = images[:, :, 0:3, :, :]
        B, S, C, H, W = images.shape
        images = images.view(B * S, C, H, W)
        return F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

    def get_dino_feats_map(self, images):
        B, S, _, H, W = images.shape
        flat = images[:, :, 0:3, :, :].contiguous().view(B * S, 3, H, W)
        feature_maps = self.feature_backend.extract(flat, target_device=images.device)
        _, C, feat_h, feat_w = feature_maps.shape
        return feature_maps.view(B, S, C, feat_h, feat_w)
        
       
    def get_unet_feats_map(self, images):
        B, S, C, H, W = images.shape
        if self.unet is None:
            raise RuntimeError("UNet backbone is disabled in feature_backbone_only mode.")
        images = self._prepare_images(images, self.unet_input_size)
        unet_outputs = self.unet(images)
        _, C, H, W = unet_outputs.shape

        return unet_outputs.view(B, S, C, H, W)

    def get_dift_feats_map(self, images: torch.Tensor) -> torch.Tensor:
        if not self.use_dift:
            raise RuntimeError("DIFT feature extraction is disabled. Set network.dift.enabled=True in config.")
        if dift_backend is None:
            raise ImportError(
                "DIFT backend is not available. Clone the DIFT repo (e.g., DACoN/dift) or set DACON_DIFT_PATH/FGW_DIFT_PATH."
            )

        B, S, C, H, W = images.shape
        device = images.device
        images = self._prepare_images(images, self.dift_img_size)

        dift_maps = []
        for idx in range(images.shape[0]):
            img_tensor = images[idx].detach().cpu().clamp(0.0, 1.0)
            pil_img = to_pil_image(img_tensor)
            feature_map = dift_backend.extract_dift_features(
                img=pil_img,
                img_size=self.dift_img_size,
                t=self.dift_t,
                up_ft_index=self.dift_up_ft_index,
                ensemble_size=self.dift_ensemble_size,
                prompt=self.dift_prompt,
                cache_dir=self.dift_cache_dir,
                device=self.dift_device,
                model_id=self.dift_model_id,
            )
            if not isinstance(feature_map, torch.Tensor):
                feature_map = torch.from_numpy(feature_map)
            feature_map = feature_map.to(device)
            dift_maps.append(feature_map)

        dift_maps_tensor = torch.stack(dift_maps, dim=0)
        target_h, target_w = self.segment_pool_size
        if dift_maps_tensor.shape[-2:] != (target_h, target_w):
            dift_maps_tensor = F.interpolate(
                dift_maps_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        dift_maps_tensor = dift_maps_tensor.view(B, S, *dift_maps_tensor.shape[1:])
        return dift_maps_tensor

    def dino_dim_reduction(self, seg_dino_feats):
        B, S, L, C = seg_dino_feats.shape
        seg_dino_feats = self.dino_mlp(seg_dino_feats.view(B*S*L, C))

        return seg_dino_feats.view(B, S, L, -1)
    
    def dino_unet_fusion(self, seg_dino_feats, seg_unet_feats):
        B, S, L, C = seg_dino_feats.shape
        dino_flat_feats = seg_dino_feats.view(B*S*L, C)
        unet_flat_feats = seg_unet_feats.view(B*S*L, C)

        if self.version == "1_0":
            seg_fusion_feats = self.dino_unet_mlp(torch.cat([dino_flat_feats, unet_flat_feats], dim=-1))
        else:
            seg_fusion_feats = self.l2_normalize(dino_flat_feats) + self.l2_normalize(unet_flat_feats)
            
        return seg_fusion_feats.view(B, S, L, -1)

    def get_segment_feats(self, feats_map, seg_image, seg_num):

        if self.version == "1_0":
            _, _, H, W = seg_image.shape
            seg_feats_src = segment_pooling(feats_map, seg_image, seg_num, (H,W))
        else:
            seg_feats_src = segment_pooling(feats_map, seg_image, seg_num, self.segment_pool_size)
 
        return  seg_feats_src


    def get_seg_cos_sim(self, seg_feats_src, seg_feats_tgt):
        
        seg_feats_src = F.normalize(seg_feats_src, p=2, dim=-1)
        seg_feats_tgt = F.normalize(seg_feats_tgt, p=2, dim=-1)

        B, S_src, L_src, C = seg_feats_src.shape
        seg_feats_src = seg_feats_src.view(B, S_src*L_src, C)

        B, S_tgt, L_tgt, C = seg_feats_tgt.shape
        seg_feats_tgt = seg_feats_tgt.view(B, S_tgt*L_tgt, C)
        
        seg_cos_sim = torch.matmul(seg_feats_tgt, seg_feats_src.transpose(-1, -2))

        return seg_cos_sim
    
    def _process_single(self, line_image, seg_image, seg_num):

        line_images = line_image.unsqueeze(1)
        seg_images = seg_image.unsqueeze(1)
        seg_nums = seg_num.unsqueeze(1)

        seg_feats, raw_dino_feats, seg_dift_feats = self._process_multi(line_images, seg_images, seg_nums)

        if seg_dift_feats is not None:
            seg_dift_feats = seg_dift_feats.squeeze(1)
        return seg_feats.squeeze(1), raw_dino_feats.squeeze(1), seg_dift_feats
    
    def _process_multi(self, line_images, seg_images, seg_nums):

        dino_feats_map = self.get_dino_feats_map(line_images)
        seg_dino_feats = self.get_segment_feats(dino_feats_map, seg_images, seg_nums)
        raw_dino_feats = seg_dino_feats.clone() 

        seg_unet_feats = None
        if not self.feature_backbone_only:
            unet_feats_map = self.get_unet_feats_map(line_images)
            seg_unet_feats = self.get_segment_feats(unet_feats_map, seg_images, seg_nums)

        seg_dift_feats: Optional[torch.Tensor] = None
        dift_feats_map: Optional[torch.Tensor] = None
        if self.use_dift:
            dift_feats_map = self.get_dift_feats_map(line_images)
            seg_dift_feats = self.get_segment_feats(dift_feats_map, seg_images, seg_nums)
            self.latest_dift_feats_map = dift_feats_map.detach()
            self.latest_seg_dift_feats = seg_dift_feats.detach()
        else:
            self.latest_dift_feats_map = None
            self.latest_seg_dift_feats = None

        if self.feature_backbone_only:
            seg_feats = self.l2_normalize(seg_dino_feats)
        else:
            seg_dino_feats_reduced = self.dino_dim_reduction(seg_dino_feats)
            seg_feats = self.dino_unet_fusion(seg_dino_feats_reduced, seg_unet_feats)

        return seg_feats, raw_dino_feats, seg_dift_feats

    def forward(self, data):

        seg_feats_src, seg_dino_feats_src, seg_dift_feats_src = self._process_multi(
            data['line_images_src'], data['seg_images_src'], data["seg_nums_src"]
        )
        seg_feats_tgt, seg_dino_feats_tgt, seg_dift_feats_tgt = self._process_multi(
            data['line_images_tgt'], data['seg_images_tgt'], data["seg_nums_tgt"]
        )

        dino_seg_cos_sim = self.get_seg_cos_sim(seg_dino_feats_src, seg_dino_feats_tgt)
        seg_cos_sim = self.get_seg_cos_sim(seg_feats_src, seg_feats_tgt)

        dift_seg_cos_sim: Optional[torch.Tensor] = None
        if self.use_dift and seg_dift_feats_src is not None and seg_dift_feats_tgt is not None:
            dift_seg_cos_sim = self.get_seg_cos_sim(seg_dift_feats_src, seg_dift_feats_tgt)

        self.latest_seg_dift_feats_src = (
            seg_dift_feats_src.detach() if seg_dift_feats_src is not None else None
        )
        self.latest_seg_dift_feats_tgt = (
            seg_dift_feats_tgt.detach() if seg_dift_feats_tgt is not None else None
        )
        self.latest_dift_sim_map = dift_seg_cos_sim

        return seg_cos_sim, dino_seg_cos_sim

    def compute_dift_similarity(self, data) -> torch.Tensor:
        if not self.use_dift:
            raise RuntimeError("DIFT feature extraction is disabled. Enable it via network.dift.enabled in the config.")

        self.forward(data)

        if self.latest_dift_sim_map is None:
            raise RuntimeError("DIFT similarity map was not computed. Check that DIFT features are available.")

        return self.latest_dift_sim_map



