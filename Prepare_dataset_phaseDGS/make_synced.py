"""
make_synced.py — Synced condition.

Cuts the same source frame range [min_frames, max_frames) from every camera.

Output: <output_dir>/<scene>/synced/{<camera>.mp4, info/frame_map.json, info/summary.txt}

Usage:
  python make_synced.py --scene selcap_bike --min_frames 2000 --max_frames 2300
  python make_synced.py --scene nvidia-jumping            # whole video
"""

import argparse
from pathlib import Path

import utils


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    utils.add_common_args(parser)
    args = parser.parse_args()

    video_paths = utils.get_video_list(Path(args.source_dir) / args.scene)
    output_dir = Path(args.output_dir) / args.scene / "synced"
    infos = {vp.stem: utils.get_video_info(vp) for vp in video_paths}
    start, end = utils.resolve_frame_range(infos, args.min_frames, args.max_frames)
    num_frames = end - start
    if num_frames <= 0:
        raise SystemExit(f"[Error] empty frame range [{start}, {end})")

    source_indices = list(range(start, end))
    for vp in video_paths:
        utils.extract_and_write(vp, source_indices, output_dir / f"{vp.stem}.mp4", infos[vp.stem])

    frame_map = {str(i): src for i, src in enumerate(source_indices)}
    utils.save_json(
        {
            "scene": args.scene,
            "mode": "synced",
            "output_frame_count": num_frames,
            "videos": {vp.stem: {"src_start": start, "frame_map": frame_map} for vp in video_paths},
        },
        output_dir / "info" / "frame_map.json",
    )

    lines = [
        "=== Synced ===",
        f"Scene         : {args.scene}",
        f"Source range  : [{start}, {end})",
        f"Output frames : {num_frames}",
        "",
        f"{'Camera':<10} | {'Src frames':>10} | {'FPS':>6} | {'Resolution':>12}",
    ]
    for cam, info in infos.items():
        lines.append(f"{cam:<10} | {info['frame_count']:>10} | {info['fps']:>6.2f} | {info['width']}x{info['height']:>6}")
    utils.save_text("\n".join(lines) + "\n", output_dir / "info" / "summary.txt")


if __name__ == "__main__":
    main()
