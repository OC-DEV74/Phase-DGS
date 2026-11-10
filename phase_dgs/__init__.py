"""Phase-DGS: phase extraction module, feature bank and losses."""

from phase_dgs.checkpoint import (
    FEATURE_BANK_FILE,
    PHASE_PROJECTOR_FILE,
    ROI_FILE,
    SYNC_CONFIG_FILE,
    SYNC_MODEL_DIR,
    load_trained_config,
    load_trained_extractor,
    load_trained_feature_bank,
)
from phase_dgs.config import SyncConfig
from phase_dgs.extractor import (
    PhaseExtractor,
    PhaseProjector,
    Roi,
    crop_to_vit_input,
    extract_all_features,
    load_roi_camera_names,
    load_rois,
)
from phase_dgs.feature_bank import FeatureBank, init_feature_bank, upsample_features
from phase_dgs.inference import camera_phases, load_frame_map, reference_phases, smooth_phases
from phase_dgs.losses import monotonic_loss, phase_consistency_loss
from phase_dgs.phase_sync import PhaseSync

__all__ = [
    "FEATURE_BANK_FILE",
    "PHASE_PROJECTOR_FILE",
    "ROI_FILE",
    "SYNC_CONFIG_FILE",
    "SYNC_MODEL_DIR",
    "FeatureBank",
    "PhaseExtractor",
    "PhaseProjector",
    "PhaseSync",
    "Roi",
    "SyncConfig",
    "camera_phases",
    "crop_to_vit_input",
    "extract_all_features",
    "init_feature_bank",
    "load_frame_map",
    "load_roi_camera_names",
    "load_rois",
    "load_trained_config",
    "load_trained_extractor",
    "load_trained_feature_bank",
    "monotonic_loss",
    "phase_consistency_loss",
    "reference_phases",
    "smooth_phases",
    "upsample_features",
]
