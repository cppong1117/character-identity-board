"""Capture real UI screenshots of the live Character Identity Board frontend.

These are genuine headless-Chromium renders of the running system (real API data),
used as V0.1 evidence. Produces PNGs into ~/character-identity-board/evidence/V0.1/.
"""
from __future__ import annotations
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

EVID = Path.home() / "character-identity-board" / "evidence" / "V0.1"
EVID.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8322"


def shot(page, name):
    path = EVID / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print("saved", path)
    return path


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})

    # 1) Projects page
    pg.goto(BASE + "/#view=projects", wait_until="networkidle")
    time.sleep(1.5)
    shot(pg, "ui_projects")

    # 2) Project detail (open project 9)
    pg.goto(BASE + "/#view=project:9", wait_until="networkidle")
    time.sleep(2.0)
    shot(pg, "ui_project_detail")

    # 3) Character continuity (project 9)
    pg.goto(BASE + "/#view=character:8:9", wait_until="networkidle")
    time.sleep(2.0)
    shot(pg, "ui_character_continuity")

    # 4) Shot board detail (shot 31)
    pg.goto(BASE + "/#view=shot:31:9", wait_until="networkidle")
    time.sleep(2.0)
    shot(pg, "ui_shot_detail")

    b.close()
print("done")

# (appended isn't run; via cli below)
