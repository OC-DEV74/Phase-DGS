"""Shared helpers of the dataset generators (make_synced / make_shifted / make_dropped)."""

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def get_video_list(scene_dir: Path) -> list[Path]:
    """Video files directly inside ``scene_dir``, sorted by name (= camera order)."""
    videos = sorted(
        p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in: {scene_dir}")
    return videos


def get_video_info(video_path: Path) -> dict:
    """frame_count, fps, width and height of a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def extract_and_write(video_path: Path, src_indices: list[int], out_path: Path, info: dict):
    """Writes the source frames ``src_indices`` of a video, in order, to ``out_path``.

    Frames are streamed; an unreadable frame is replaced by the previous one.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = info["width"], info["height"]
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (width, height)
    )
    cap = cv2.VideoCapture(str(video_path))

    last_frame = np.zeros((height, width, 3), dtype=np.uint8)
    prev_idx = None
    for idx in tqdm(src_indices, desc=f"  {video_path.stem}", unit="f", leave=False):
        if prev_idx is None or idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            last_frame = frame
        writer.write(last_frame)
        prev_idx = idx

    cap.release()
    writer.release()


def resolve_frame_range(video_infos: dict[str, dict], start: int, end: int) -> tuple[int, int]:
    """``end = -1`` means the frame count of the shortest video."""
    if end == -1:
        end = min(v["frame_count"] for v in video_infos.values())
        print(f"[Info] end frame set to {end} (shortest video)")
    return start, end


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Save] {path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[Save] {path}")


def add_common_args(parser):
    parser.add_argument("--scene", required=True, help="scene name (folder in --source_dir)")
    parser.add_argument("--source_dir", default="./source_video", help="root of the source videos")
    parser.add_argument("--output_dir", default="./output", help="root of the generated datasets")
    parser.add_argument("--min_frames", type=int, default=0, help="first source frame")
    parser.add_argument("--max_frames", type=int, default=-1, help="end source frame (exclusive, -1 = shortest video)")
