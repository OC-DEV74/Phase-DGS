"""Phase-DGS losses (paper Sec. 3.4)."""

import torch
import torch.nn.functional as F
from torch import Tensor

from phase_dgs.extractor import PhaseProjector


def phase_consistency_loss(
    features: Tensor,
    phases: Tensor,
    bank_features: Tensor,
    projector: PhaseProjector,
) -> Tensor:
    """Phase consistency loss L_phase (Eq. 4-6).

    Args:
        features:      (B, d) features z of the input images.
        phases:        (B, 1) phases p of the input images.
        bank_features: (B, F*k, d) feature bank rows B_i of each image's camera.
        projector:     phase projector.
    """
    phase_map = projector(bank_features)  # (B, F*k, 1)
    weights = torch.softmax(-(phase_map - phases[:, None, :]).abs(), dim=1)
    ref_features = (weights * bank_features).sum(dim=1)  # (B, d)

    return F.mse_loss(features, ref_features)


def monotonic_loss(bank_features: Tensor, projector: PhaseProjector) -> Tensor:
    """Monotonic loss L_mono (Eq. 7-8).

    Penalizes adjacent phase map entries whose increment is below 1/(F*k).

    Args:
        bank_features: (I, F*k, d) the whole feature bank.
        projector:     phase projector.
    """
    phase_map = projector(bank_features).squeeze(-1)  # (I, F*k)
    delta = phase_map[:, 1:] - phase_map[:, :-1]
    min_increment = 1 / phase_map.size(1)

    return torch.relu(min_increment - delta).sum()
