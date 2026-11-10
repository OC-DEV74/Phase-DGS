"""
make_shifted.py — Shifted condition.

Each camera starts at a random offset in [0, max_offset] frames. All output
videos have the same length: max_frames - (min_frames + largest offset).

Output: <output_dir>/<scene>/shifted/{<camera>.mp4, info/frame_map.json, info/summary.txt}

Usage:
  python make_shifted.py --scene nvidia-jumping --max_offset 20 --seed 42
  python make_shifted.py --scene selcap_bike --min_frames 2000 --max_frames 2330 --max_offset 30
"""

import argparse
import random
from pathlib import Path

import utils


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    utils.add_common_args(parser)
    parser.add_argument("--max_offset", type=int, default=20, help="largest start offset in frames")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    video_paths = utils.get_video_list(Path(args.source_dir) / args.scene)
    output_dir = Path(args.output_dir) / args.scene / "shifted"
    infos = {vp.stem: utils.get_video_info(vp) for vp in video_paths}
    start, end = utils.resolve_frame_range(infos, args.min_frames, args.max_frames)

    offsets = {cam: random.randint(0, args.max_offset) for cam in infos}
    src_starts = {cam: start + offset for cam, offset in offsets.items()}
    num_frames = end - max(src_starts.values())
    if num_frames <= 0:
        raise SystemExit("[Error] no frames left, reduce --max_offset or widen the frame range")
    print(f"[Info] offsets: {offsets}, output frames: {num_frames}")

    videos = {}
    for vp in video_paths:
        cam = vp.stem
        source_indices = list(range(src_starts[cam], src_starts[cam] + num_frames))
        utils.extract_and_write(vp, source_indices, output_dir / f"{cam}.mp4", infos[cam])
        videos[cam] = {
            "offset": offsets[cam],
            "src_start": src_starts[cam],
            "frame_map": {str(i): src for i, src in enumerate(source_indices)},
        }

    utils.save_json(
        {"scene": args.scene, "mode": "shifted", "output_frame_count": num_frames, "videos": videos},
        output_dir / "info" / "frame_map.json",
    )

    lines = [
        "=== Shifted ===",
        f"Scene         : {args.scene}",
        f"Source range  : [{start}, {end})",
        f"Output frames : {num_frames}",
        f"Seed          : {args.seed if args.seed is not None else 'random'}",
        "",
        f"{'Camera':<10} | {'Offset':>6} | {'Src start':>9} | {'Src end':>7}",
    ]
    for cam, offset in offsets.items():
        lines.append(f"{cam:<10} | {offset:>6} | {src_starts[cam]:>9} | {src_starts[cam] + num_frames - 1:>7}")
    utils.save_text("\n".join(lines) + "\n", output_dir / "info" / "summary.txt")


if __name__ == "__main__":
    main()
