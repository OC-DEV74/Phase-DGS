"""Phase-DGS settings."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Self


@dataclass
class SyncConfig:
    """Phase-DGS settings. Naming follows the paper (Sec. 3, 4.1).

    Settings of the paper: ``FB_upsample_factor = 2`` with FreeTimeGS, and
    ``FB_upsample_factor = 3``, ``lambda_mono = 0.01`` with RealTime-4DGS.
    """

    roi_json: Path  # per-camera ROIs from tools/select_roi.py
    dinov2_model: str = "dinov2_vitb14"  # ViT encoder (dinov2_vits14 / vitb14 / vitl14)
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])  # phase projector hidden layers
    lambda_phase: float = 0.1  # phase consistency loss weight
    lambda_mono: float = 0.1  # monotonic loss weight
    lr_phase_projector: float = 1e-4  # phase projector learning rate
    lr_feature_bank: float = 1e-3  # feature bank learning rate
    init_feature_bank: bool = True  # initialize the feature bank from the train frames
    FB_upsample_factor: int = 1  # upsampling factor k: feature bank length = num_frames * k
    ratio_CLS: float = 0.0  # feature = ratio_CLS * CLS token + (1 - ratio_CLS) * mean patch token
    enabled: bool = True  # use Phase-DGS; read by your training loop, not by phase_dgs (false: timestamps)

    def __post_init__(self):
        self.roi_json = Path(self.roi_json).expanduser()

    @classmethod
    def from_toml(cls, path: str | Path, table: str = "sync") -> Self:
        """Reads the ``[sync]`` table of a TOML config (see ``configs/sync.toml``)."""
        with open(path, "rb") as f:
            data = tomllib.load(f)[table]

        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown [{table}] keys in {path}: {sorted(unknown)}")
        return cls(**data)
