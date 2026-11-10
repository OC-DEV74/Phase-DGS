"""Phases at test time.

- ``reference_phases``: phase of each test-view frame for evaluation (paper
  App. B, Algorithm 1). It needs the frame map ``info/frame_map.json`` written
  by ``Prepare_dataset_phaseDGS``.
- ``camera_phases`` + ``smooth_phases``: phases of one reference camera, used
  as the global phase for rendering a video (paper App. C).
"""

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from phase_dgs.extractor import PhaseExtractor, extract_all_features

logger = logging.getLogger(__name__)


def load_frame_map(path: str | Path) -> dict[str, dict[int, int]]:
    """Per-camera map from video frame index to source frame index."""
    with open(path, "r") as f:
        videos = json.load(f)["videos"]

    return {
        name: {int(frame): int(source) for frame, source in video["frame_map"].items()}
        for name, video in videos.items()
    }


@torch.no_grad()
def reference_phases(
    *,
    extractor: PhaseExtractor,
    rgb: Tensor,
    test_frames: Sequence[tuple[int, int]],
    train_cameras: Sequence[int],
    camera_names: Sequence[str],
    frame_map: dict[str, dict[int, int]],
    first_frame: int = 0,
    fallback_times: Sequence[float] | None = None,
) -> list[float]:
    """Reference phase of each test frame (Algorithm 1).

    The reference phase is the mean phase of all train-view frames sampled from
    the same source frame as the test frame.

    Args:
        extractor:      trained phase extractor.
        rgb:            (F, C, H, W, 3) uint8 frames of all cameras; ``rgb[f, c]`` is
                        video frame ``first_frame + f`` of absolute camera ``c``.
        test_frames:    (camera, video frame) of each test image.
        train_cameras:  absolute indices of the train cameras.
        camera_names:   name (video stem) of each absolute camera index, e.g.
                        ``load_roi_camera_names(roi_json)``.
        frame_map:      output of ``load_frame_map``.
        first_frame:    video frame index of ``rgb[0]``.
        fallback_times: time used for a test frame whose source frame is not
                        in any train view (e.g. its timestamp). Without it, such
                        a frame raises ``ValueError``.
    """
    missing = set(camera_names) - frame_map.keys()
    if missing:
        raise ValueError(f"cameras missing from the frame map: {sorted(missing)}")

    source_to_frame = {
        name: {source: frame for frame, source in frame_map[name].items()}
        for name in camera_names
    }
    num_frames = rgb.size(0)

    phases = []
    for i, (camera, frame) in enumerate(test_frames):
        source = frame_map[camera_names[camera]].get(frame)

        images, cameras = [], []
        for train_camera in train_cameras:
            train_frame = source_to_frame[camera_names[train_camera]].get(source)  # type: ignore[arg-type]
            if train_frame is not None and 0 <= train_frame - first_frame < num_frames:
                images.append(rgb[train_frame - first_frame, train_camera])
                cameras.append(train_camera)

        if not images:
            if fallback_times is None:
                raise ValueError(
                    f"no train frame shares the source frame of camera {camera_names[camera]} "
                    f"frame {frame}; pass fallback_times"
                )
            logger.warning(
                "no train frame shares the source frame of camera %s frame %d, "
                "falling back to its timestamp",
                camera_names[camera],
                frame,
            )
            phases.append(float(fallback_times[i]))
            continue

        train_phases, _ = extractor(torch.stack(images), cameras)
        phases.append(float(train_phases.mean()))

    return phases


@torch.no_grad()
def camera_phases(extractor: PhaseExtractor, rgb: Tensor, camera: int) -> list[float]:
    """Phase of every frame of one camera; ``rgb`` is its (F, H, W, 3) uint8 frames."""
    features = extract_all_features(extractor, rgb.unsqueeze(1), [camera])  # (1, F, d)

    return extractor.projector(features).flatten().tolist()


def smooth_phases(values: Sequence[float], window: int = 25) -> list[float]:
    """Moving average with reflect padding; returns the input if it is too short."""
    if window < 2 or len(values) < window:
        return list(values)

    # window - 1 padded values in total, so the output has the input length
    padded = np.pad(np.array(values), (window // 2, (window - 1) // 2), mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid").tolist()
