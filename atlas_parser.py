"""
ATLAS Data Parser Module

Custom parser for MITRE ATLAS data since the mitreattack-python library
only supports ATT&CK Enterprise/ICS/Mobile matrices.
"""

import json
from typing import Dict, List, Optional
import pandas as pd


class ATLASData:
    """Parser for MITRE ATLAS STIX 2.1 data"""

    def __init__(self, stix_filepath: str):
        """
        Initialize the ATLAS data parser.

        Args:
            stix_filepath: Path to stix-atlas.json
        """
        with open(stix_filepath, 'r') as f:
            self.stix_data = json.load(f)

        self._index_objects()

    def _index_objects(self):
        """Index STIX objects by type for quick lookup"""
        self.techniques = {}
        self.tactics = {}
        self.mitigations = {}
        self.relationships = []

        for obj in self.stix_data.get("objects", []):
            obj_type = obj.get("type")

            if obj_type == "attack-pattern":
                external_id = self._get_external_id(obj)
                if external_id:
                    self.techniques[external_id] = obj

            elif obj_type == "x-mitre-tactic":
                external_id = self._get_external_id(obj)
                if external_id:
                    self.tactics[external_id] = obj

            elif obj_type == "course-of-action":
                external_id = self._get_external_id(obj)
                if external_id:
                    self.mitigations[external_id] = obj

            elif obj_type == "relationship":
                self.relationships.append(obj)

    def _get_external_id(self, obj: Dict) -> Optional[str]:
        """Extract ATLAS external ID from STIX object"""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-atlas":
                return ref.get("external_id")
        return None

    def get_techniques(self, include_subtechniques: bool = True) -> List[Dict]:
        """
        Get all techniques as a list of dicts.

        Args:
            include_subtechniques: Whether to include subtechniques (default True)

        Returns:
            List of technique dictionaries
        """
        result = []
        for ext_id, obj in self.techniques.items():
            is_subtechnique = obj.get("x_mitre_is_subtechnique", False)
            if not include_subtechniques and is_subtechnique:
                continue
            result.append({
                "id": obj["id"],
                "external_id": ext_id,
                "name": obj["name"],
                "description": obj.get("description", ""),
                "kill_chain_phases": obj.get("kill_chain_phases", []),
                "is_subtechnique": is_subtechnique
            })
        return result

    def get_tactics(self) -> List[Dict]:
        """Get all tactics ordered by their typical sequence"""
        # Define the standard ATLAS tactic order
        tactic_order = [
            "AML.TA0002",  # Reconnaissance
            "AML.TA0003",  # Resource Development
            "AML.TA0004",  # Initial Access
            "AML.TA0000",  # AI Model Access
            "AML.TA0005",  # Execution
            "AML.TA0006",  # Persistence
            "AML.TA0012",  # Privilege Escalation
            "AML.TA0007",  # Defense Evasion
            "AML.TA0013",  # Credential Access
            "AML.TA0008",  # Discovery
            "AML.TA0015",  # Lateral Movement
            "AML.TA0009",  # Collection
            "AML.TA0001",  # AI Attack Staging
            "AML.TA0014",  # Command and Control
            "AML.TA0010",  # Exfiltration
            "AML.TA0011",  # Impact
        ]

        result = []
        for tactic_id in tactic_order:
            if tactic_id in self.tactics:
                obj = self.tactics[tactic_id]
                result.append({
                    "id": obj["id"],
                    "external_id": tactic_id,
                    "name": obj["name"],
                    "shortname": obj.get("x_mitre_shortname", ""),
                    "description": obj.get("description", "")
                })
        return result

    def get_tactic_names_ordered(self) -> List[str]:
        """Get tactic names in the standard kill chain order"""
        return [t["name"] for t in self.get_tactics()]

    def get_technique_by_id(self, technique_id: str) -> Optional[Dict]:
        """Get a specific technique by its ATLAS ID (e.g., AML.T0051)"""
        return self.techniques.get(technique_id)

    def get_atlas_id(self, stix_id: str) -> Optional[str]:
        """Get ATLAS ID from STIX ID"""
        for ext_id, obj in self.techniques.items():
            if obj["id"] == stix_id:
                return ext_id
        return None

    def get_tactic_name_by_shortname(self, shortname: str) -> str:
        """Convert tactic shortname to full name"""
        for tactic in self.tactics.values():
            if tactic.get("x_mitre_shortname") == shortname:
                return tactic["name"]
        return shortname

    def get_tactic_name_by_id(self, tactic_id: str) -> str:
        """Convert tactic external ID (e.g., AML.TA0002) to full name"""
        if tactic_id in self.tactics:
            return self.tactics[tactic_id]["name"]
        return tactic_id

    def get_techniques_for_tactic(self, tactic_shortname: str) -> List[Dict]:
        """Get all techniques belonging to a specific tactic"""
        result = []
        for ext_id, obj in self.techniques.items():
            kill_chain_phases = obj.get("kill_chain_phases", [])
            for phase in kill_chain_phases:
                if phase.get("phase_name") == tactic_shortname:
                    result.append({
                        "id": obj["id"],
                        "external_id": ext_id,
                        "name": obj["name"],
                        "description": obj.get("description", ""),
                        "is_subtechnique": obj.get("x_mitre_is_subtechnique", False)
                    })
                    break
        return result


def load_atlas_case_studies(filepath: str) -> pd.DataFrame:
    """
    Load ATLAS case studies from JSON file.

    Args:
        filepath: Path to atlas-case-studies.json

    Returns:
        DataFrame with case study data
    """
    return pd.read_json(filepath)


def get_techniques_from_case_study_procedure(
    procedure: List[Dict],
    atlas_data: ATLASData
) -> List[Dict]:
    """
    Extract technique details from a case study procedure.

    Args:
        procedure: List of procedure steps from case study
        atlas_data: ATLASData instance for technique lookup

    Returns:
        List of technique details with tactic context
    """
    techniques_used = []
    for step in procedure:
        technique_id = step.get("technique")
        tech_obj = atlas_data.get_technique_by_id(technique_id)
        if tech_obj:
            # Get the tactic name from the step (using external ID like AML.TA0002)
            tactic_id = step.get("tactic", "")
            if tactic_id:
                # Convert tactic ID to full name
                tactic_name = atlas_data.get_tactic_name_by_id(tactic_id)
            else:
                # Try to get from technique's kill chain phases
                kill_chain_phases = tech_obj.get("kill_chain_phases", [])
                if kill_chain_phases:
                    tactic_name = atlas_data.get_tactic_name_by_shortname(
                        kill_chain_phases[0].get("phase_name", "Unknown")
                    )
                else:
                    tactic_name = "Unknown"

            techniques_used.append({
                "Technique Name": tech_obj["name"],
                "ATT&CK ID": technique_id,
                "Phase Name": tactic_name,
                "Description": step.get("description", tech_obj.get("description", ""))
            })
    return techniques_used
