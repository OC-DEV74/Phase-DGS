"""Feature bank (paper Sec. 3.3 and App. A)."""

import torch
from torch import Tensor, nn


class FeatureBank(nn.Module):
    """Learnable per-camera reference features B in R^{I x (F*k) x d}.

    I is the number of (train) cameras, F the number of frames, k the
    upsampling factor and d the feature dimension.
    """

    def __init__(self, num_cameras: int, num_frames: int, feature_dim: int, upsample_factor: int):
        super().__init__()

        self.upsample_factor = upsample_factor
        self.features = nn.Parameter(
            torch.randn(num_cameras, num_frames * upsample_factor, feature_dim)
        )

    @property
    def length(self) -> int:
        return self.features.size(1)

    def forward(self, cameras: Tensor | None = None) -> Tensor:
        """Features of the given bank cameras, (len(cameras), F*k, d), or of all cameras."""
        if cameras is None:
            return self.features
        return self.features[cameras]


def upsample_features(features: Tensor, factor: int) -> Tensor:
    """Upsamples (C, F, d) per-frame features to (C, F*factor, d).

    Frame f is placed at slot f*factor and the slots in between are filled by
    linear interpolation of the neighboring frames. Slots after the last frame
    repeat it.
    """
    num_frames = features.size(1)

    positions = torch.arange(num_frames * factor, device=features.device) / factor
    positions = positions.clamp(max=num_frames - 1)
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=num_frames - 1)
    weight = (positions - lower)[None, :, None]

    return features[:, lower] * (1 - weight) + features[:, upper] * weight


@torch.no_grad()
def init_feature_bank(bank: FeatureBank, frame_features: Tensor):
    """Initializes the bank from the (I, F, d) features of all train frames."""
    bank.features.copy_(upsample_features(frame_features, bank.upsample_factor))
