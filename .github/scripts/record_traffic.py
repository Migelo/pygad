#!/usr/bin/env python3
"""Merge GitHub traffic data into a long-term JSON log.

Reads freshly-fetched clones/views JSON (written by the workflow into /tmp)
and merges the per-day figures into .github/traffic/traffic.json, keyed by
date (YYYY-MM-DD). Each run backfills the last 14 days, so a missed run is
automatically recovered on the next one.
"""
import json
import sys
from pathlib import Path

TRAFFIC_FILE = Path(".github/traffic/traffic.json")

# API field name -> (count key, unique-count key) we want to store.
KEYMAP = {
    "clones": ("clones", "unique_cloners"),
    "views": ("views", "unique_viewers"),
}


def load_log() -> dict:
    if not TRAFFIC_FILE.exists():
        return {}
    try:
        return json.loads(TRAFFIC_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def main() -> int:
    log = load_log()

    for kind, (count_key, uniq_key) in KEYMAP.items():
        raw_path = Path(f"/tmp/{kind}.json")
        if not raw_path.exists():
            print(f"warning: {raw_path} not found, skipping '{kind}'")
            continue
        try:
            raw = json.loads(raw_path.read_text())
        except json.JSONDecodeError as exc:
            print(f"error: could not parse {raw_path}: {exc}")
            continue
        for entry in raw.get(kind, []):
            date = entry["timestamp"][:10]  # YYYY-MM-DD
            day = log.setdefault(date, {"date": date})
            day[count_key] = entry.get("count", 0)
            day[uniq_key] = entry.get("uniques", 0)

    if not log:
        print("no data to write")
        return 1

    TRAFFIC_FILE.parent.mkdir(parents=True, exist_ok=True)
    ordered = {d: log[d] for d in sorted(log)}
    TRAFFIC_FILE.write_text(json.dumps(ordered, indent=2) + "\n")

    latest = max(ordered)
    n = len(ordered)
    total_clones = sum(d.get("clones", 0) for d in ordered.values())
    total_views = sum(d.get("views", 0) for d in ordered.values())
    print(f"recorded traffic: {n} days | "
          f"latest={latest} | "
          f"total clones={total_clones} views={total_views}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
