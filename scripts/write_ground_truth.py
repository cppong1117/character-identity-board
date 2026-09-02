"""Generate Redistributable benchmark metadata and lightweight ground truth.

Ground truth is explicit for generated fixtures: every six-second hard-cut segment
has one expected identity token. The clips are real H.264 video files.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.home() / "character-identity-board-data"
OUT = ROOT / "benchmarks" / "V0.1"
GEN = OUT / "generated"


def main() -> None:
    cases = [
        {
            "id": "A",
            "video": "testA_two_person_hardcuts.mp4",
            "description": "two-person hard-cut conversation fixture",
            "shot_ground_truth": ["A", "B", "A", "B"],
            "expected_shots": 4,
            "checks": ["shot_boundary", "cross_shot_identity", "manual_reassign", "shot_board"],
        },
        {
            "id": "B",
            "video": "testB_lowlight_back_and_forth.mp4",
            "description": "low-light degraded fixture; low-quality observations should enter review",
            "shot_ground_truth": ["A", "B", "A", "B"],
            "expected_shots": 4,
            "checks": ["unknown", "low_confidence", "review_queue"],
        },
        {
            "id": "C",
            "video": "testC_three_person_clusters.mp4",
            "description": "multi-shot discovery fixture used for merge/split/export checks",
            "shot_ground_truth": ["A", "B", "A", "B", "A", "B"],
            "expected_shots": 6,
            "checks": ["discovery", "merge", "split", "rename", "export"],
        },
    ]
    manifest = {"dataset": "CIB V0.1 generated benchmark", "cases": cases}
    (OUT / "GROUND_TRUTH.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
