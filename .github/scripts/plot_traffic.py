#!/usr/bin/env python3
"""Plot GitHub traffic history from .github/traffic/traffic.json.

Usage:
    .venv/bin/python .github/scripts/plot_traffic.py [--out traffic.png] [--days 0]

--days N restricts the plot to the last N days (0 = all).
Writes a PNG (Agg backend, no display needed) and prints summary totals.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

TRAFFIC_FILE = Path(__file__).resolve().parent.parent / "traffic" / "traffic.json"


def load_traffic(path=TRAFFIC_FILE):
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        sys.exit(f"error: {path} not found")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: could not parse {path}: {exc}")
    days = [raw[d] for d in sorted(raw)]
    return days


def summarize(days):
    return {
        "days": len(days),
        "first": days[0]["date"],
        "last": days[-1]["date"],
        "clones": sum(d.get("clones", 0) for d in days),
        "unique_cloners": sum(d.get("unique_cloners", 0) for d in days),
        "views": sum(d.get("views", 0) for d in days),
        "unique_viewers": sum(d.get("unique_viewers", 0) for d in days),
    }


def main():
    ap = argparse.ArgumentParser(description="Plot GitHub traffic history.")
    ap.add_argument("--out", default="traffic.png", help="output PNG path")
    ap.add_argument("--days", type=int, default=0,
                    help="plot only the last N days (0 = all)")
    ap.add_argument("--traffic-file", default=TRAFFIC_FILE,
                    help="input traffic.json path")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = load_traffic(args.traffic_file)
    if args.days > 0:
        days = days[-args.days :]
    if not days:
        sys.exit("error: no traffic data")

    total = summarize(days)
    print(
        f"{total['days']} days {total['first']}..{total['last']} | "
        f"clones={total['clones']} (uniq {total['unique_cloners']}) | "
        f"views={total['views']} (uniq {total['unique_viewers']})"
    )

    xs = [date.fromisoformat(d["date"]) for d in days]
    clones = [d.get("clones", 0) for d in days]
    uclones = [d.get("unique_cloners", 0) for d in days]
    views = [d.get("views", 0) for d in days]
    uviews = [d.get("unique_viewers", 0) for d in days]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"GitHub traffic ({total['first']}..{total['last']})")

    ax1.bar(xs, clones, label="clones")
    ax1.plot(xs, uclones, marker="o", ms=2, label="unique cloners")
    ax1.set_ylabel("clones / day")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.bar(xs, views, label="views", color="C1")
    ax2.plot(xs, uviews, marker="o", ms=2, color="C3",
             label="unique viewers")
    ax2.set_ylabel("views / day")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=100)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
