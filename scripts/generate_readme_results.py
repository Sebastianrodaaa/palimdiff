#!/usr/bin/env python3
"""Generate results table markdown from results/*.json (no invented numbers)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    results_dir = Path("results")
    files = sorted(results_dir.glob("*.json"))
    if not files:
        print("<!-- No results/*.json yet -->")
        return
    latest = files[-1]
    data = json.loads(latest.read_text())
    lines = [
        f"<!-- AUTO-GENERATED from `{latest}` -->",
        "",
        "| Arm | Mean denoised tokens (retention subset) | Note |",
        "|-----|----------------------------------------|------|",
    ]
    from collections import defaultdict

    sums: dict[str, list[int]] = defaultdict(list)
    for row in data.get("retention_probe", []):
        sums[row["strategy"]].append(row["denoised_tokens_total"])
    for arm, vals in sorted(sums.items()):
        mean = sum(vals) / len(vals) if vals else 0
        lines.append(f"| `{arm}` | {mean:.1f} | from real tiny-stand-in run |")
    lines.append("")
    lines.append(f"_Source: `{latest.name}`, denoiser=`{data.get('denoiser')}`._")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
