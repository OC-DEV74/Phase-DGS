"""Attaches Phase-DGS to the training loop of a 4DGS backbone (paper Sec. 3.5, 3.6)."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from phase_dgs.checkpoint import FEATURE_BANK_FILE, PHASE_PROJECTOR_FILE, ROI_FILE, SYNC_CONFIG_FILE
from phase_dgs.config import SyncConfig
from phase_dgs.extractor import (
    PhaseExtractor,
    extract_all_features,
    load_roi_camera_names,
    load_rois,
)
from phase_dgs.feature_bank import FeatureBank, init_feature_bank
from phase_dgs.losses import monotonic_loss, phase_consistency_loss


class PhaseSync:
    """Phase-DGS components attached to the 4DGS training loop (paper Sec. 3.6).

    The phase extracted from each train image replaces the timestamp t of the
    backbone, and L_phase + L_mono are added to the rendering loss.

    Args:
        config:        Phase-DGS settings.
        train_rgb:     (F, I, H, W, 3) uint8 frames of the I train cameras; it
                       may stay on the CPU (only used to initialize the feature bank).
        train_cameras: absolute camera index of each column of ``train_rgb``, i.e.
                       its index in the sorted video names of ``roi.json``.
        scale:         resolution of the images that will be fed, relative to the
                       original video resolution the ROIs were selected on.
        device:        device of the Phase-DGS modules.
    """

    def __init__(
        self,
        config: SyncConfig,
        train_rgb: Tensor,
        train_cameras: Sequence[int],
        scale: float,
        device: str | torch.device = "cuda",
    ):
        self.config = config
        self.scale = scale

        rois, input_size = load_rois(config.roi_json, scale)
        self.camera_names = load_roi_camera_names(config.roi_json)
        self.extractor = PhaseExtractor(
            config.dinov2_model, config.hidden_dims, rois, input_size, config.ratio_CLS
        ).to(device)

        # feature bank row i belongs to the i-th train camera
        self.train_cameras = list(train_cameras)
        self.bank_rows = {camera: row for row, camera in enumerate(self.train_cameras)}

        num_frames, num_cameras = train_rgb.shape[:2]
        if num_cameras != len(self.train_cameras):
            raise ValueError(
                f"train_rgb has {num_cameras} cameras but {len(self.train_cameras)} train_cameras were given"
            )
        self.bank = FeatureBank(
            num_cameras,
            num_frames,
            self.extractor.feature_dim,
            config.FB_upsample_factor,
        ).to(device)
        if config.init_feature_bank:
            init_feature_bank(
                self.bank,
                extract_all_features(self.extractor, train_rgb, self.train_cameras),
            )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.extractor.projector.parameters(), "lr": config.lr_phase_projector},
                {"params": self.bank.parameters(), "lr": config.lr_feature_bank},
            ],
            eps=1e-15,
        )

    def phases(self, images: Tensor, cameras: Sequence[int]) -> tuple[Tensor, Tensor]:
        """(B, 1) phases and (B, d) features of the (B, H, W, 3) uint8 train images."""
        return self.extractor(images, cameras)

    def losses(
        self, features: Tensor, phases: Tensor, cameras: Sequence[int]
    ) -> dict[str, Tensor]:
        """Weighted L_phase and L_mono."""
        rows = torch.tensor([self.bank_rows[c] for c in cameras], device=features.device)
        projector = self.extractor.projector

        return {
            "phase": self.config.lambda_phase
            * phase_consistency_loss(features, phases, self.bank(rows), projector),
            "mono": self.config.lambda_mono * monotonic_loss(self.bank(), projector),
        }

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def save(self, directory: str | Path):
        """Saves the trained modules and their settings.

        Use ``directory = <run>/model_sync`` for ``phase_dgs.checkpoint``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.extractor.save_projector(directory / PHASE_PROJECTOR_FILE)
        torch.save(self.bank.state_dict(), directory / FEATURE_BANK_FILE)

        # the projector only works with the ROIs and settings it was trained with
        roi_copy = directory / ROI_FILE
        if not (roi_copy.exists() and roi_copy.samefile(self.config.roi_json)):
            shutil.copyfile(self.config.roi_json, roi_copy)
        # the saved roi_json names the copy above, so no local path is written
        config = asdict(self.config) | {"roi_json": ROI_FILE}
        with open(directory / SYNC_CONFIG_FILE, "w") as f:
            json.dump({"config": config, "scale": self.scale}, f, indent=2)

    def plot(self, rgb: Tensor, cameras: Sequence[int], path: str | Path, title: str = ""):
        """Phase graph of the feature bank and of the cameras in ``rgb``.

        Args:
            rgb:     (F, C, H, W, 3) uint8 frames.
            cameras: absolute camera index of each column of ``rgb``.
        """
        # imported here: phase_dgs.plot switches matplotlib to the Agg backend
        from phase_dgs.plot import plot_phase_graph

        plot_phase_graph(
            extractor=self.extractor,
            bank=self.bank,
            bank_camera_names=[self.camera_names[c] for c in self.train_cameras],
            rgb=rgb,
            cameras=cameras,
            camera_names=[self.camera_names[c] for c in cameras],
            path=path,
            title=title,
        )
