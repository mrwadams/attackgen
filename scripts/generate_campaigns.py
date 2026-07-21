#!/usr/bin/env python3
"""Regenerate data/campaigns.json and data/campaigns_ics.json from the bundled STIX.

Page 1 offers a "Campaign" source that presents a dropdown built from these files
(name -> attack.mitre.org URL) and then looks the campaign's techniques up in the
STIX bundle at runtime. Pre-generating the dropdown data keeps it cheap — the
53 MB Enterprise bundle stays lazily loaded and is only read when a scenario is
actually generated (the same reason data/groups.json exists).

So whenever the ATT&CK bundles in data/ are refreshed, rerun this to keep the
campaign dropdowns in sync with the shipped version — otherwise campaigns added
in a new ATT&CK release never appear.

Usage:
    python scripts/generate_campaigns.py
"""

import json
import sys
from pathlib import Path

from mitreattack.stix20 import MitreAttackData

DATA_DIR = Path(__file__).parent.parent / "data"

# (STIX bundle in data/, output mapping file)
SOURCES = [
    ("enterprise-attack.json", "campaigns.json"),
    ("ics-attack.json", "campaigns_ics.json"),
]


def _attack_url(campaign) -> str | None:
    """The campaign's canonical attack.mitre.org URL from its ATT&CK reference."""
    for ref in campaign.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("url")
    return None


def build_rows(bundle_path: Path) -> list[dict]:
    data = MitreAttackData(str(bundle_path))
    rows = []
    for campaign in data.get_campaigns(remove_revoked_deprecated=True):
        url = _attack_url(campaign)
        if url:
            # Reuse the "group" key so the page/loader code stays uniform with
            # the threat-group dropdown (data/groups.json).
            rows.append({"group": campaign["name"], "url": url})
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
        print(f"Wrote {len(rows)} campaigns to data/{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
