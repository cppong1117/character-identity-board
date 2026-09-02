"""Create or reset an isolated benchmark SQLite database for repeated API E2E runs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path.home() / "character-identity-board-data"
DB = ROOT / "benchmarks" / "V0.1" / "benchmark.sqlite3"
DB.parent.mkdir(parents=True, exist_ok=True)
if DB.exists():
    DB.unlink()
# This script is used by shell wrappers before importing the app.
print(DB)
