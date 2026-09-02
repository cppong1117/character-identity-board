"""Create local benchmark videos from bundled, redistributable test images.

This produces real video files (not static-image tests): each clip is encoded as
an H.264 video with four hard-cut shots, controlled motion and zoom.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path.home() / "character-identity-board-data"
OUT = ROOT / "benchmarks" / "V0.1" / "generated"
FACE1 = ROOT / "cache" / "test_face.jpg"
FACE2 = ROOT / "cache" / "test_face2.jpg"


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def make_clip(name: str, images: list[Path], *, low_light=False, multi=False) -> Path:
    out = OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{name}.parts"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    parts = []
    for i, image in enumerate(images):
        part = tmp / f"shot_{i:02d}.mp4"
        # Use a real encoded segment per shot; concat then preserves hard cuts.
        vf = "scale=854:480,zoompan=z='1+0.00025*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=854x480:fps=25"
        if low_light:
            vf += ",eq=brightness=-0.22:contrast=0.82"
        if multi and i == 1:
            vf += ",hflip"
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(image),
             "-vf", vf, "-t", "6", "-r", "25", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "18", str(part)])
        parts.append(part)
    listing = tmp / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)])
    shutil.rmtree(tmp)
    return out


def main() -> None:
    if not FACE1.exists() or not FACE2.exists():
        raise SystemExit("Expected local test_face.jpg and test_face2.jpg are missing")
    make_clip("testA_two_person_hardcuts.mp4", [FACE1, FACE2, FACE1, FACE2])
    make_clip("testB_lowlight_back_and_forth.mp4", [FACE1, FACE2, FACE1, FACE2], low_light=True)
    make_clip("testC_three_person_clusters.mp4", [FACE1, FACE2, FACE1, FACE2, FACE1, FACE2], multi=True)
    print(f"generated benchmark videos in {OUT}")


if __name__ == "__main__":
    main()
