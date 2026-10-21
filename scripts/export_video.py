"""Export saved navigation RGB frames to an MP4 without starting Habitat."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "ffmpeg was not found. Install it in the active environment, or run navigation with --no-video."
        )
    return executable


def export_video(episode_dir: Path, fps: float = 10.0, output: Path = None) -> Path:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive")
    executable = find_ffmpeg()
    episode_dir = episode_dir.resolve()
    frames = sorted((episode_dir / "rgb").glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not frames:
        raise FileNotFoundError(f"No numbered RGB frames found in {episode_dir / 'rgb'}")
    if any(frame.name != f"{index:06d}.png" for index, frame in enumerate(frames)):
        raise ValueError("RGB frames must be consecutive, starting at 000000.png")
    output = output.resolve() if output is not None else episode_dir / "trajectory.mp4"
    if output.exists():
        raise FileExistsError(f"Video already exists: {output}. Select a new --output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Encode to a temporary file, so a failed export leaves no partial MP4.
    with tempfile.TemporaryDirectory(prefix=".video-", dir=output.parent) as temporary:
        encoded = Path(temporary) / "trajectory.mp4"
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-framerate", str(fps), "-start_number", "0",
            "-i", str(episode_dir / "rgb" / "%06d.png"),
            "-frames:v", str(len(frames)), "-an", "-c:v", "libx264",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Video export failed; saved navigation data is intact:\n{result.stderr.strip()}")
        encoded.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, help="Default: <episode-dir>/trajectory.mp4")
    args = parser.parse_args()
    output = export_video(args.episode_dir, args.fps, args.output)
    print(f"Video written to: {output}")


if __name__ == "__main__":
    main()
