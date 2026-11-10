"""
make_full_frame_roi.py — Writes a roi.json whose ROIs cover each whole frame (no cropping).

Use it instead of tools/select_roi.py to train without the ROI crop (paper Sec. 4.5).

Usage:
    python tools/make_full_frame_roi.py datasets/<scene>/                  # -> datasets/<scene>/roi.json
    python tools/make_full_frame_roi.py datasets/<scene>/ -o roi.json --input-size 224
"""

import argparse
import json
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def write_full_frame_rois(video_dir: Path, out_path: Path, input_size: int = 224):
    """Writes a roi.json whose ROIs cover each whole frame (no cropping)."""
    rois = {}
    for video in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        rois[video.name] = {"x": 0, "y": 0, "width": width, "height": height}

    if not rois:
        raise SystemExit(f"no videos found in {video_dir}")

    with open(out_path, "w") as f:
        json.dump({"input_size": input_size, "rois": rois}, f, indent=2)
    print(f"[roi] wrote full-frame ROIs for {len(rois)} cameras to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_dir", type=Path, help="folder with one video per camera")
    parser.add_argument("--output", "-o", type=Path, default=None, help="output path (default: <video_dir>/roi.json)")
    parser.add_argument("--input-size", type=int, default=224, help="ViT input size (default: 224)")
    args = parser.parse_args()

    write_full_frame_rois(args.video_dir, args.output or args.video_dir / "roi.json", args.input_size)


if __name__ == "__main__":
    main()
