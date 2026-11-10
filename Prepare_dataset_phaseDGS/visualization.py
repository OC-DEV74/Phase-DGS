"""
visualization.py — Inspect the generated datasets of a scene (synced / shifted / dropped).

--plot_frame_map:
  Plots output frame index vs. source frame index of every camera from
  info/frame_map.json into info/frame_map_plot.png.

--grid_video:
  Tiles all camera videos into an N x M grid video, info/grid_video.mp4.
  Cells are resized so that the grid fits in UHD (3840x2160).

Usage:
  python visualization.py --scene nvidia-jumping --plot_frame_map
  python visualization.py --scene nvidia-jumping --grid_video --grid_cols 4
  python visualization.py --scene nvidia-jumping --plot_frame_map --grid_video
"""

import argparse
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils

UHD_W, UHD_H = 3840, 2160


def find_mode_dirs(scene_dir: Path) -> list[Path]:
    """Sub-directories of output/<scene>/ that contain info/frame_map.json."""
    modes = sorted(d for d in scene_dir.iterdir() if (d / "info" / "frame_map.json").exists())
    if not modes:
        raise FileNotFoundError(f"frame_map.json not found: {scene_dir}/*/info/frame_map.json")
    return modes


def plot_frame_map(mode_dir: Path):
    data = utils.load_json(mode_dir / "info" / "frame_map.json")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.get_cmap("tab20").colors
    for i, (cam, video) in enumerate(data["videos"].items()):
        frame_map = video["frame_map"]
        ax.plot([int(k) for k in frame_map], list(frame_map.values()), label=cam, color=colors[i % len(colors)], linewidth=1.4)

    ax.set_xlabel("Output frame index", fontsize=12)
    ax.set_ylabel("Source frame index", fontsize=12)
    ax.set_title(f"Frame map  -  scene: {data['scene']}  |  mode: {data.get('mode', mode_dir.name)}", fontsize=14)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = mode_dir / "info" / "frame_map_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] {out_path}")


def cell_size(src_w: int, src_h: int, n_cols: int, n_rows: int) -> tuple[int, int]:
    """Cell size such that the grid fits in UHD, keeping the aspect ratio (never upscales)."""
    scale = min(UHD_W // n_cols / src_w, UHD_H // n_rows / src_h, 1.0)
    cw, ch = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    return cw - cw % 2, ch - ch % 2  # even sizes for the video encoder


def put_label(image: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_grid_video(mode_dir: Path, n_cols: int | None = None, n_rows: int | None = None):
    video_paths = utils.get_video_list(mode_dir)
    n_cams = len(video_paths)
    if n_cols is None and n_rows is None:
        n_cols = math.ceil(math.sqrt(n_cams))
    if n_cols is None:
        n_cols = math.ceil(n_cams / n_rows)
    if n_rows is None:
        n_rows = math.ceil(n_cams / n_cols)

    info = utils.get_video_info(video_paths[0])
    total_frames = info["frame_count"]
    cw, ch = cell_size(info["width"], info["height"], n_cols, n_rows)
    grid_w, grid_h = cw * n_cols, ch * n_rows
    print(f"[Grid] {n_cams} cameras -> {n_rows}x{n_cols}, cell {cw}x{ch}, {total_frames} frames")

    videos = utils.load_json(mode_dir / "info" / "frame_map.json").get("videos", {})
    frame_maps = [videos.get(vp.stem, {}).get("frame_map", {}) for vp in video_paths]

    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    out_path = mode_dir / "info" / "grid_video.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (grid_w, grid_h))
    blank = np.zeros((ch, cw, 3), dtype=np.uint8)

    for frame_idx in range(total_frames):
        cells = []
        for cap in caps:
            ret, frame = cap.read()
            cells.append(cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA) if ret else blank.copy())
        cells += [blank.copy()] * (n_rows * n_cols - len(cells))
        grid = np.vstack([np.hstack(cells[r * n_cols : (r + 1) * n_cols]) for r in range(n_rows)])

        for i, vp in enumerate(video_paths):
            r, c = divmod(i, n_cols)
            put_label(grid, vp.stem, (c * cw + 6, r * ch + 26), 0.6, (255, 255, 255))
            source = frame_maps[i].get(str(frame_idx), "?")
            put_label(grid, f"out:{frame_idx}  src:{source}", (c * cw + 6, r * ch + 50), 0.5, (255, 255, 0))

        counter = f"{frame_idx + 1} / {total_frames}"
        (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        put_label(grid, counter, (grid_w - tw - 8, grid_h - 8), 0.6, (255, 255, 255))
        writer.write(grid)

    writer.release()
    for cap in caps:
        cap.release()
    print(f"[Grid] {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--output_dir", default="./output", help="root of the generated datasets")
    parser.add_argument("--plot_frame_map", action="store_true", help="plot info/frame_map.json -> info/frame_map_plot.png")
    parser.add_argument("--grid_video", action="store_true", help="tile all cameras -> info/grid_video.mp4")
    parser.add_argument("--grid_cols", type=int, default=None, help="grid columns (default: automatic)")
    parser.add_argument("--grid_rows", type=int, default=None, help="grid rows (default: automatic)")
    args = parser.parse_args()

    if not args.plot_frame_map and not args.grid_video:
        raise SystemExit("nothing to do, pass --plot_frame_map and/or --grid_video")

    for mode_dir in find_mode_dirs(Path(args.output_dir) / args.scene):
        print(f"\n[Mode] {mode_dir.name}")
        if args.plot_frame_map:
            plot_frame_map(mode_dir)
        if args.grid_video:
            make_grid_video(mode_dir, n_cols=args.grid_cols, n_rows=args.grid_rows)


if __name__ == "__main__":
    main()
