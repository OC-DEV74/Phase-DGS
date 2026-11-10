"""Phase graph: phase map of the feature bank and per-frame phases of each camera."""

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from phase_dgs.extractor import PhaseExtractor, extract_all_features
from phase_dgs.feature_bank import FeatureBank


@torch.no_grad()
def plot_phase_graph(
    *,
    extractor: PhaseExtractor,
    bank: FeatureBank,
    bank_camera_names: Sequence[str],
    rgb: Tensor,
    cameras: Sequence[int],
    camera_names: Sequence[str],
    path: str | Path,
    title: str = "",
):
    """Saves a two-panel phase graph.

    Args:
        extractor:         phase extraction module.
        bank:              feature bank.
        bank_camera_names: name of each feature bank row (train cameras).
        rgb:               (F, C, H, W, 3) uint8 images of the plotted cameras.
        cameras:           absolute camera index of each column of ``rgb``.
        camera_names:      name of each column of ``rgb``.
        path:              output image path.
        title:             suffix of the panel titles.
    """
    phase_map = extractor.projector(bank()).squeeze(-1).cpu()  # (I, F*k)
    phases = extractor.projector(extract_all_features(extractor, rgb, cameras))
    phases = phases.squeeze(-1).cpu()  # (C, F)

    fig, (ax_bank, ax_frames) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, curves, names, xlabel, panel in [
        (ax_bank, phase_map, bank_camera_names, "Feature bank index", "Feature bank phase map"),
        (ax_frames, phases, camera_names, "Frame index", "Per-frame phase"),
    ]:
        cmap = plt.get_cmap("tab20", len(curves))
        for i, (curve, name) in enumerate(zip(curves, names, strict=True)):
            ax.plot(curve, marker=".", markersize=2, linewidth=1.0, color=cmap(i), label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Phase")
        ax.set_title(f"{panel} {title}".strip())
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
