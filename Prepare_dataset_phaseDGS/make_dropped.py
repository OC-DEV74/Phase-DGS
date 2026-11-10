"""
make_dropped.py — Dropped condition.

Combines, per camera (as set in config/<scene>_config.json):
  - offset:      random start offset in [0, max_offset]
  - frame drop:  removes frame_drop_rate of the frames, one by one
  - block drop:  removes frame_block_drop_rate of the frames in consecutive blocks of
                 frame_block_min_size..frame_block_max_size frames
Dropped frames are removed (not duplicated); all videos are trimmed to the
shortest resulting length.

If the scene config does not exist, a default one is written and the script
stops so it can be edited.

Output: <output_dir>/<scene>/dropped/{<camera>.mp4, info/frame_map.json, info/summary.txt}

Usage:
  python make_dropped.py --scene nvidia-jumping --seed 42
"""

import argparse
import random
from pathlib import Path

import utils

CONFIG_DIR = Path(__file__).parent / "config"

DEFAULT_GLOBAL = {
    "min_frames": 0,
    "max_frames": -1,  # -1 = shortest video
    "max_offset": 30,
    "frame_drop_rate": 0.05,
    "frame_block_drop_rate": 0.10,
    "frame_block_min_size": 5,
    "frame_block_max_size": 30,
}
DEFAULT_CAMERA = {"enable_offset": True, "enable_frame_drop": False, "enable_frame_block": False}


def load_config(path: Path, scene: str, cameras: list[str]) -> dict:
    if not path.exists():
        utils.save_json(
            {
                "scene": scene,
                "global": dict(DEFAULT_GLOBAL),
                "cameras": {cam: dict(DEFAULT_CAMERA) for cam in cameras},
            },
            path,
        )
        raise SystemExit(f"[Config] wrote a default config to {path.resolve()}; edit it and run again")

    config = utils.load_json(path)
    for cam in cameras:
        config["cameras"].setdefault(cam, dict(DEFAULT_CAMERA))
    return config


def drop_frames(indices: list[int], drop_rate: float) -> tuple[list[int], list[int]]:
    """Removes ``drop_rate`` of the frames at random. Returns (kept, dropped source frames)."""
    n = len(indices)
    n_drop = min(max(0, int(round(n * drop_rate))), n - 1)  # keep at least one frame
    dropped = set(random.sample(range(n), n_drop))

    kept = [idx for i, idx in enumerate(indices) if i not in dropped]
    return kept, [indices[i] for i in sorted(dropped)]


def drop_blocks(
    indices: list[int], drop_rate: float, min_size: int, max_size: int
) -> tuple[list[int], list[list[int]]]:
    """Removes about ``drop_rate`` of the frames in non-overlapping consecutive blocks.

    Returns (kept, dropped blocks as [first source frame, last source frame]).
    """
    n = len(indices)
    target = max(0, int(round(n * drop_rate)))
    keep = [True] * n
    blocks = []
    total = 0

    for _ in range(200):  # attempts
        if total >= target:
            break
        size_max = min(max_size, target - total, n - 2)
        if size_max < min_size:
            break

        size = random.randint(min_size, size_max)
        start = random.randint(0, n - size - 1)
        end = start + size - 1
        if not all(keep[start : end + 1]):
            continue  # overlaps a dropped block

        keep[start : end + 1] = [False] * size
        blocks.append([indices[start], indices[end]])
        total += size

    return [idx for i, idx in enumerate(indices) if keep[i]], blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene name (folder in --source_dir)")
    parser.add_argument("--source_dir", default="./source_video", help="root of the source videos")
    parser.add_argument("--output_dir", default="./output", help="root of the generated datasets")
    parser.add_argument("--config", type=Path, default=None, help="scene config (default: config/<scene>_config.json)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    video_paths = utils.get_video_list(Path(args.source_dir) / args.scene)
    cameras = [vp.stem for vp in video_paths]
    output_dir = Path(args.output_dir) / args.scene / "dropped"

    config = load_config(args.config or CONFIG_DIR / f"{args.scene}_config.json", args.scene, cameras)
    g = config["global"]
    infos = {vp.stem: utils.get_video_info(vp) for vp in video_paths}
    start, end = utils.resolve_frame_range(infos, g["min_frames"], g["max_frames"])

    offsets = {
        cam: random.randint(0, min(g["max_offset"], infos[cam]["frame_count"] - 1))
        if config["cameras"][cam].get("enable_offset", False)
        else 0
        for cam in cameras
    }
    src_starts = {cam: start + offsets[cam] for cam in cameras}
    base_frames = end - max(src_starts.values())
    if base_frames <= 0:
        raise SystemExit("[Error] no frames left, reduce max_offset or widen the frame range")

    # pass 1: source frame list of each camera
    sources = {}
    for cam in cameras:
        cam_config = config["cameras"][cam]
        indices = list(range(src_starts[cam], src_starts[cam] + base_frames))
        dropped, blocks = [], []
        if cam_config.get("enable_frame_drop", False):
            indices, dropped = drop_frames(indices, g["frame_drop_rate"])
        if cam_config.get("enable_frame_block", False):
            indices, blocks = drop_blocks(
                indices, g["frame_block_drop_rate"], g["frame_block_min_size"], g["frame_block_max_size"]
            )
        sources[cam] = {"indices": indices, "dropped": dropped, "blocks": blocks}

    # pass 2: trim to the shortest camera and write
    num_frames = min(len(s["indices"]) for s in sources.values())
    print(f"[Info] base frames: {base_frames}, output frames: {num_frames}")

    videos = {}
    for vp in video_paths:
        cam = vp.stem
        indices = sources[cam]["indices"][:num_frames]
        utils.extract_and_write(vp, indices, output_dir / f"{cam}.mp4", infos[cam])
        videos[cam] = {
            "offset": offsets[cam],
            "src_start": src_starts[cam],
            "base_frames": base_frames,
            "actual_frames": num_frames,
            "dropped_src": sources[cam]["dropped"],
            "blocked_src": sources[cam]["blocks"],
            "frame_map": {str(i): src for i, src in enumerate(indices)},
        }

    utils.save_json(
        {
            "scene": args.scene,
            "mode": "dropped",
            "base_frame_count": base_frames,
            "final_frame_count": num_frames,
            "videos": videos,
        },
        output_dir / "info" / "frame_map.json",
    )

    lines = [
        "=== Dropped ===",
        f"Scene         : {args.scene}",
        f"Output frames : {num_frames}",
        f"Seed          : {args.seed if args.seed is not None else 'random'}",
        "",
        f"{'Camera':<10} | {'Offset':>6} | {'Drop':>4} | {'Block':>5} | Details",
    ]
    for cam in cameras:
        cam_config, s = config["cameras"][cam], sources[cam]
        details = []
        if s["dropped"]:
            details.append(f"dropped {len(s['dropped'])} frames")
        if s["blocks"]:
            details.append(f"{len(s['blocks'])} blocks")
        lines.append(
            f"{cam:<10} | {offsets[cam]:>6} | {'ON' if cam_config.get('enable_frame_drop') else 'OFF':>4} | "
            f"{'ON' if cam_config.get('enable_frame_block') else 'OFF':>5} | {', '.join(details) or '-'}"
        )
    utils.save_text("\n".join(lines) + "\n", output_dir / "info" / "summary.txt")


if __name__ == "__main__":
    main()
