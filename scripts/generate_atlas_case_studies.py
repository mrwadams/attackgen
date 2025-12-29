#!/usr/bin/env python3
"""
Script to generate atlas-case-studies.json from ATLAS.yaml

Downloads the latest ATLAS.yaml from the mitre-atlas/atlas-data repository
and converts case studies to a JSON format compatible with AttackGen's
existing groups.json structure.

Run this periodically to update case study data.
"""

import json
import yaml
import urllib.request
import sys
from pathlib import Path

ATLAS_YAML_URL = "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "atlas-case-studies.json"


def fetch_atlas_yaml() -> dict:
    """Fetch ATLAS.yaml from GitHub"""
    print(f"Fetching ATLAS.yaml from {ATLAS_YAML_URL}...")
    with urllib.request.urlopen(ATLAS_YAML_URL) as response:
        content = response.read().decode('utf-8')
    return yaml.safe_load(content)


def convert_case_studies(atlas_data: dict) -> list:
    """
    Convert ATLAS case studies to AttackGen-compatible format.

    Target format matches groups.json structure:
    {
        "group": "case study name",  # Used for display and selection
        "url": "https://atlas.mitre.org/studies/ID",
        "id": "AML.CS0000",  # Case study ID
        "procedure": [...],  # Attack procedure steps
        "target": "...",
        "actor": "...",
        "summary": "...",
        "case_study_type": "incident|exercise"
    }
    """
    case_studies = []

    for cs in atlas_data.get("case-studies", []):
        cs_id = cs.get("id", "")

        # Convert procedure steps
        procedure = []
        for step in cs.get("procedure", []):
            procedure.append({
                "tactic": step.get("tactic", ""),
                "technique": step.get("technique", ""),
                "description": step.get("description", "")
            })

        case_study = {
            "group": cs.get("name", "Unknown Case Study"),  # 'group' key for UI compatibility
            "id": cs_id,
            "url": f"https://atlas.mitre.org/studies/{cs_id}",
            "procedure": procedure,
            "target": cs.get("target", ""),
            "actor": cs.get("actor", "Unknown"),
            "summary": cs.get("summary", ""),
            "case_study_type": cs.get("case-study-type", "incident")
        }
        case_studies.append(case_study)

    # Sort by ID for consistent ordering
    case_studies.sort(key=lambda x: x["id"])

    return case_studies


def main():
    try:
        # Fetch and parse ATLAS.yaml
        atlas_data = fetch_atlas_yaml()

        # Convert case studies
        case_studies = convert_case_studies(atlas_data)

        # Ensure output directory exists
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(case_studies, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(case_studies)} case studies to {OUTPUT_FILE}")

        # Print summary
        incident_count = sum(1 for cs in case_studies if cs["case_study_type"] == "incident")
        exercise_count = len(case_studies) - incident_count
        print(f"  - Incidents: {incident_count}")
        print(f"  - Exercises: {exercise_count}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
