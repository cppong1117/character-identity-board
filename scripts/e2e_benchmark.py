"""Benchmark API E2E runner: upload/process/query/export/correction persistence."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path.home() / "character-identity-board-data"
VIDEO = ROOT / "benchmarks" / "V0.1" / "generated" / "testA_two_person_hardcuts.mp4"

def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.main import app
    client = TestClient(app)
    report = {"checks": [], "video": str(VIDEO)}

    r = client.post("/projects", json={"name": "E2E Test A"})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    report["checks"].append("create_project")

    with VIDEO.open("rb") as fh:
        r = client.post(f"/projects/{pid}/videos", files={"file": (VIDEO.name, fh, "video/mp4")})
    assert r.status_code == 201, r.text
    vid = r.json()["id"]
    report["checks"].append("upload_video")

    r = client.post(f"/projects/{pid}/videos/process", json={})
    assert r.status_code == 200, r.text
    for _ in range(90):
        time.sleep(1)
        status = client.get(f"/projects/{pid}/videos/{vid}/status").json()
        if status["status"] in {"completed", "failed"}:
            break
    assert status["status"] == "completed", status
    report["checks"].append("pipeline_completed")

    shots = client.get(f"/projects/{pid}/shots")
    assert shots.status_code == 200 and len(shots.json()) >= 4, shots.text
    report["shot_count"] = len(shots.json())
    report["checks"].append("shots")

    chars = client.get(f"/projects/{pid}/characters")
    assert chars.status_code == 200 and len(chars.json()) >= 1, chars.text
    report["character_count"] = len(chars.json())
    report["checks"].append("characters")

    queue = client.get(f"/projects/{pid}/review-queue")
    assert queue.status_code == 200
    report["review_queue_count"] = len(queue.json())
    report["checks"].append("review_queue")

    # Rename + tracklet reassignment persistence path.
    c0 = chars.json()[0]
    renamed = client.patch(f"/characters/{c0['id']}", json={"display_name": "Character A"})
    assert renamed.status_code == 200, renamed.text
    report["checks"].append("rename_character")

    # All tracklets have an assignment in this fixture; confirm one if present.
    tr = client.get(f"/shots/{shots.json()[0]['id']}/tracklets")
    assert tr.status_code == 200
    if tr.json():
        tid = tr.json()[0]["id"]
        assign = client.patch(f"/tracklets/{tid}/assignment", json={"character_id": c0["id"], "review_status": "confirmed"})
        assert assign.status_code == 200, assign.text
        report["checks"].append("manual_reassign_confirm")

    exp = client.post(f"/projects/{pid}/export", json={"formats": ["json", "csv", "contact_sheet", "html"]})
    assert exp.status_code == 200, exp.text
    manifest = exp.json()
    report["exports"] = manifest
    report["checks"].append("exports")

    # Verify persistence after the same app's DB session is reopened by a new client.
    p2 = TestClient(app).get(f"/projects/{pid}")
    assert p2.status_code == 200
    assert p2.json()["name"] == "E2E Test A"
    report["checks"].append("restart-equivalent-read")

    out = ROOT / "benchmarks" / "V0.1" / "e2e_test_a_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
