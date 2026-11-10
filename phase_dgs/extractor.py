"""Phase extraction module (paper Sec. 3.2).

An image is cropped to its camera's ROI, encoded by a frozen DINOv2 ViT, and
mapped to a phase p in [0, 1] by the phase projector.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import Normalize

IMAGENET_NORMALIZE = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@dataclass(frozen=True)
class Roi:
    """Motion-intensive region of interest of one camera, in pixels."""

    x: int
    y: int
    width: int
    height: int

    def scaled(self, scale: float) -> Self:
        return type(self)(
            x=int(self.x * scale),
            y=int(self.y * scale),
            width=int(self.width * scale),
            height=int(self.height * scale),
        )


def load_rois(path: str | Path, scale: float = 1.0) -> tuple[list[Roi], int]:
    """Loads ``roi.json`` written by ``tools/select_roi.py``.

    Returns the per-camera ROIs, ordered by video file name (which is also the
    camera index order), scaled to the training image resolution, together
    with the ViT input size.
    """
    with open(path, "r") as f:
        data = json.load(f)

    rois = [Roi(**data["rois"][name]).scaled(scale) for name in sorted(data["rois"])]
    return rois, int(data["input_size"])


def load_roi_camera_names(path: str | Path) -> list[str]:
    """Camera names (video file stems) of ``roi.json``, in camera index order."""
    with open(path, "r") as f:
        data = json.load(f)

    return [Path(name).stem for name in sorted(data["rois"])]


def crop_to_vit_input(images: Tensor, rois: Sequence[Roi], size: int) -> Tensor:
    """Crops each image to its ROI and resizes it to the ViT input resolution.

    Args:
        images: (B, H, W, 3) uint8 images.
        rois:   one ROI per image.
        size:   ViT input resolution.

    Returns:
        (B, 3, size, size) ImageNet-normalized float tensor.
    """
    crops = []
    for image, roi in zip(images, rois, strict=True):
        crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        crop = crop.permute(2, 0, 1).unsqueeze(0).float() / 255
        crops.append(
            F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
        )

    return IMAGENET_NORMALIZE(torch.cat(crops))


class PhaseProjector(nn.Module):
    """MLP mapping a feature z to a phase p in [0, 1] (Eq. 2).

    Architecture: ``(Linear -> GELU) x len(hidden_dims) -> Linear -> Sigmoid``.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (256, 128, 64)):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.GELU()]
            prev_dim = hidden_dim
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """(..., d) features -> (..., 1) phases."""
        return self.mlp(features)


class PhaseExtractor(nn.Module):
    """ROI crop + frozen DINOv2 encoder + phase projector."""

    def __init__(
        self,
        dinov2_model: str,
        hidden_dims: Sequence[int],
        rois: Sequence[Roi],
        input_size: int,
        ratio_CLS: float = 0.0,
    ):
        super().__init__()

        self.vit: nn.Module = torch.hub.load("facebookresearch/dinov2", dinov2_model)  # type: ignore[assignment]
        self.vit.eval().requires_grad_(False)
        self.projector = PhaseProjector(self.feature_dim, hidden_dims)

        self.rois = list(rois)
        self.input_size = input_size
        self.ratio_CLS = ratio_CLS

    @property
    def feature_dim(self) -> int:
        return self.vit.embed_dim  # type: ignore[return-value]

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.vit.eval()  # the encoder stays frozen
        return self

    @torch.no_grad()
    def extract_features(self, images: Tensor, cameras: Sequence[int]) -> Tensor:
        """(B, H, W, 3) uint8 images of the given cameras -> (B, d) features z.

        The feature is a blend of the DINOv2 CLS token and the mean of the patch
        tokens: ``ratio_CLS * CLS + (1 - ratio_CLS) * mean(patch)``.
        """
        vit_input = crop_to_vit_input(
            images, [self.rois[c] for c in cameras], self.input_size
        )
        # one ViT pass gives both; x_norm_clstoken is what self.vit(...) returns
        tokens = self.vit.forward_features(vit_input.to(self.device))  # type: ignore[operator]
        CLS_token = tokens["x_norm_clstoken"]
        Patch_tokens = tokens["x_norm_patchtokens"].mean(dim=1)
        return CLS_token * self.ratio_CLS + Patch_tokens * (1 - self.ratio_CLS)

    def forward(self, images: Tensor, cameras: Sequence[int]) -> tuple[Tensor, Tensor]:
        """Returns the (B, 1) phases and the (B, d) features of the images."""
        features = self.extract_features(images, cameras)
        return self.projector(features), features

    def save_projector(self, path: str | Path):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str | Path):
        self.projector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


@torch.no_grad()
def extract_all_features(
    extractor: PhaseExtractor, rgb: Tensor, cameras: Sequence[int]
) -> Tensor:
    """Features of every frame of every camera.

    Args:
        rgb:     (F, C, H, W, 3) uint8 images; column ``j`` belongs to camera ``cameras[j]``.
        cameras: absolute camera index of each column.

    Returns:
        (C, F, d) features.
    """
    features = [extractor.extract_features(rgb[f], cameras) for f in range(rgb.size(0))]
    return torch.stack(features, dim=1)
