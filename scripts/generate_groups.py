#!/usr/bin/env python3
"""Regenerate data/groups.json and data/groups_ics.json from the bundled STIX.

The scenario pages present a threat-group dropdown built from these files
(name -> attack.mitre.org URL) and then look techniques up in the STIX bundle
at runtime. So whenever the ATT&CK bundles in data/ are refreshed, rerun this
to keep the dropdowns in sync with the shipped version — otherwise groups added
in a new ATT&CK release never appear.

Usage:
    python scripts/generate_groups.py
"""

import json
import sys
from pathlib import Path

from mitreattack.stix20 import MitreAttackData

DATA_DIR = Path(__file__).parent.parent / "data"

# (STIX bundle in data/, output mapping file)
SOURCES = [
    ("enterprise-attack.json", "groups.json"),
    ("ics-attack.json", "groups_ics.json"),
]


def _attack_url(group) -> str | None:
    """The group's canonical attack.mitre.org URL from its ATT&CK reference."""
    for ref in group.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url")
    return None


def build_rows(bundle_path: Path) -> list[dict]:
    data = MitreAttackData(str(bundle_path))
    rows = []
    for group in data.get_groups(remove_revoked_deprecated=True):
        url = _attack_url(group)
        if url:
            rows.append({"group": group["name"], "url": url})
    # Case-insensitive sort to match the existing dropdown ordering.
    rows.sort(key=lambda row: row["group"].lower())
    return rows


def main() -> int:
    for src, out in SOURCES:
        rows = build_rows(DATA_DIR / src)
        (DATA_DIR / out).write_text(
            json.dumps(rows, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} groups to data/{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
