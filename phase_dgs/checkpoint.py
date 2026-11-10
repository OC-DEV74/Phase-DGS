"""Loads the trained Phase-DGS modules of a run directory.

``PhaseSync.save`` writes them to ``<run>/model_sync/``: the weights
(phase_projector.pt, feature_bank.pt) and the settings they were trained with
(sync_config.json, roi.json).
"""

import json
from pathlib import Path

import torch

from phase_dgs.config import SyncConfig
from phase_dgs.extractor import PhaseExtractor, load_rois
from phase_dgs.feature_bank import FeatureBank

SYNC_MODEL_DIR = "model_sync"
PHASE_PROJECTOR_FILE = "phase_projector.pt"
FEATURE_BANK_FILE = "feature_bank.pt"
SYNC_CONFIG_FILE = "sync_config.json"
ROI_FILE = "roi.json"


def load_trained_config(run_path: str | Path) -> tuple[SyncConfig, float]:
    """Settings and image scale a run was trained with, as saved by ``PhaseSync.save``.

    The returned ``roi_json`` points to the copy of roi.json saved with the run.
    """
    directory = Path(run_path) / SYNC_MODEL_DIR
    with open(directory / SYNC_CONFIG_FILE, "r") as f:
        saved = json.load(f)

    config = SyncConfig(**(saved["config"] | {"roi_json": directory / ROI_FILE}))
    return config, float(saved["scale"])


def load_trained_extractor(
    config: SyncConfig, run_path: str | Path, scale: float, device: str | torch.device = "cuda"
) -> PhaseExtractor:
    """Phase extractor with the trained projector of ``run_path``.

    ``scale`` is the resolution of the images that will be fed, relative to the
    original video resolution the ROIs were selected on.
    """
    projector_path = Path(run_path) / SYNC_MODEL_DIR / PHASE_PROJECTOR_FILE
    if not projector_path.exists():
        raise FileNotFoundError(f"trained phase projector not found: {projector_path}")

    rois, input_size = load_rois(config.roi_json, scale)
    extractor = PhaseExtractor(
        config.dinov2_model, config.hidden_dims, rois, input_size, config.ratio_CLS
    )
    extractor.load_projector(projector_path)

    return extractor.to(device).eval()


def load_trained_feature_bank(run_path: str | Path, device: str | torch.device = "cuda") -> FeatureBank:
    state = torch.load(
        Path(run_path) / SYNC_MODEL_DIR / FEATURE_BANK_FILE, map_location="cpu", weights_only=True
    )
    num_cameras, length, feature_dim = state["features"].shape

    # the upsampling factor only matters for initialization
    bank = FeatureBank(num_cameras, length, feature_dim, upsample_factor=1)
    bank.load_state_dict(state)

    return bank.to(device)
