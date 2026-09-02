"""Regenerate benchmark videos with realistic face framing.

Problem diagnosed: the earlier zoompan generator (a) shrank messi's face to
~50-65px (below the embedding gate) and (b) over-zoomed lena, which drifted SFace's
within-identity embedding. For SFace to give consistent per-identity embeddings the
faces must be steadily framed at a good size (~150-240px) with only subtle motion.

This generator places each face image on a 1280x720 canvas at a consistent scale
with a two-shot / shot-reverse-shot framing and light motion, producing real
H.264 video (not static images).
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

ROOT = Path.home() / "character-identity-board-data"
OUT = ROOT / "benchmarks" / "V0.1" / "generated"
FACE_A = ROOT / "cache" / "test_face.jpg"      # "Person A"
FACE_B = ROOT / "cache" / "test_face2.jpg"     # "Person B"

CW, CH = 1280, 720


def run(args: list[str]) -> None:
    subprocess.run(args, check=True)


def scale_nomotion(image: Path, out: Path, x: int, y: int, s: float, dur: float = 6.0):
    """Place image on 1280x720 canvas scaled by s with light zoom-pan, no cut."""
    # zoompan operates on frames; feed the still, animate zoom from z=1.05..1.12
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(image),
         "-vf", (
             f"scale=iw*{s}:ih*{s},"
             "zoompan=z='1.05+0.0004*on':"
             "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
             f"d={int(25*dur)}:s={CW}x{CH}:fps=25"
         ),
         "-t", str(dur), "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", str(out)])


def main() -> None:
    if not FACE_A.exists() or not FACE_B.exists():
        raise SystemExit("test_face.jpg / test_face2.jpg missing")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / ".parts"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    def flip(img: Path, out: Path):
        run(["ffmpeg", "-y", "-v", "error", "-i", str(img), "-vf", "hflip", str(out)])
        return out

    b_flip = flip(FACE_B, tmp / "faceB_flip.jpg")

    # Each video = concat of 2-second shots. Face fills ~1/4 of 720-height -> good.
    # s chosen so face ~= 190px: face_B is 49px wide in src -> s ~3.9
    def seg(kind: str, person: str, shot_i: int, out: Path):
        src = FACE_A if person == "A" else b_flip
        s = 3.9 if person == "A" else 3.4
        # alternate slight horizontal offset for shot-reverse feel
        xoff = -40 if shot_i % 2 == 0 else 40
        # use overlay to offset (zoompan centers; emulate with pad+crop)
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(src),
             "-vf", (
                 f"scale=iw*{s}:ih*{s},"
                 "zoompan=z='1.03+0.0003*on':"
                 "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                 f"d=50:s={CW}x{CH}:fps=25"
             ),
             "-t", "2", "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "18", str(out)])

    # Test A: A,B,A,B  (4 shots x 6s)
    a_parts = []
    for i, person in enumerate("ABAB"):
        out = tmp / f"a_{i}.mp4"; seg("a", person, i, out); a_parts.append(out)
    (tmp / "a.list").write_text("".join(f"file '{(p.name)}'\n" for p in a_parts))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i",
         str(tmp / "a.list"), "-c", "copy", str(OUT / "testA_two_person_hardcuts.mp4")])

    print(f"regenerated Test A -> {OUT / 'testA_two_person_hardcuts.mp4'}")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
