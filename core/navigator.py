"""ATT&CK Navigator layer export for generated scenarios.

Every scenario is built from a concrete set of techniques (the sampled kill
chain on the Threat Group page, the multiselect on the Custom page). Those
technique IDs are already in hand at generation time, so emitting a Navigator
layer alongside the markdown download is pure serialisation — no LLM call, no
external lookup.

The layer JSON is consumed by two viewers, keyed on ``domain``:
  - Enterprise / ICS -> the MITRE ATT&CK Navigator (``enterprise-attack`` /
    ``ics-attack``).
  - ATLAS -> the ATLAS Navigator fork (``atlas-atlas``), whose layer format is
    otherwise identical except it carries no ATT&CK version.

Format verified against MITRE's own emitted layers
(github.com/mitre-atlas/atlas-navigator-data, dist/case-study-navigator-layers).
"""

from __future__ import annotations

import json
import re

# The trailing "(T1059)" / "(AML.T0051)" in a technique's display label.
_ID_IN_PARENS = re.compile(r"\(([^()]+)\)\s*$")

# Navigator layer identifiers per AttackGen matrix. A matrix absent from this
# map has no Navigator representation and yields no layer.
DOMAIN_BY_MATRIX = {
    "Enterprise": "enterprise-attack",
    "ICS": "ics-attack",
    "ATLAS": "atlas-atlas",
}

# Public Navigator UIs. Enterprise/ICS layers load in the ATT&CK Navigator;
# ATLAS layers load only in the ATLAS Navigator fork, and vice versa.
ATTACK_NAVIGATOR_URL = "https://mitre-attack.github.io/attack-navigator/"
ATLAS_NAVIGATOR_URL = "https://mitre-atlas.github.io/atlas-navigator/"

# The ATT&CK release the bundled data was cut from (data/enterprise-attack.json
# is v19.1). Bump when the STIX bundles are refreshed. Only stamped on the
# ATT&CK domains — ATLAS layers carry no `attack` version.
ATTACK_VERSION = "19"

# Layer schema version and the highlight applied to every technique in the
# scenario. Green matches AttackGen's brand accent.
LAYER_FORMAT_VERSION = "4.5"
NAVIGATOR_VERSION = "5.1.0"
HIGHLIGHT_COLOR = "#1DB954"
LEGEND_LABEL = "In this scenario"

# Type alias for the input: an ATT&CK/ATLAS technique ID paired with an
# optional tactic shortname (e.g. ("T1059", "execution") or ("T1078", None)).
Technique = tuple[str, str | None]


def parse_technique_id(display: str) -> str | None:
    """Pull the technique ID out of a ``"Name (ID)"`` display label.

    ``"Spearphishing Attachment (T1193)"`` -> ``"T1193"``,
    ``"LLM Prompt Injection (AML.T0051)"`` -> ``"AML.T0051"``. Returns ``None``
    when no trailing parenthesised ID is present.
    """
    match = _ID_IN_PARENS.search(display or "")
    return match.group(1) if match else None


def tactic_shortname(phase_name: str | None) -> str | None:
    """Convert a display phase name to a Navigator tactic shortname.

    ``"Command and Control"`` -> ``"command-and-control"``,
    ``"AI Model Access"`` -> ``"ai-model-access"``. Returns ``None`` for empty
    input so callers can omit the (optional) tactic field cleanly.
    """
    if not phase_name:
        return None
    return "-".join(phase_name.split()).lower()


def build_layer(
    *,
    name: str,
    matrix: str,
    techniques: list[Technique],
    description: str = "",
) -> dict | None:
    """Build an ATT&CK/ATLAS Navigator layer for one scenario.

    ``techniques`` is the exact set fed to the model, as ``(id, tactic)``
    pairs; ``tactic`` may be ``None`` when the source has no phase information
    (e.g. the Custom page multiselect). Duplicate IDs are collapsed, preferring
    the first occurrence that carries a tactic. The parent of any scored
    sub-technique is expanded (``showSubtechniques``) so the child is visible in
    Navigator, which otherwise collapses it out of view. Returns ``None`` for a
    matrix with no Navigator domain, so callers can skip the download button.
    """
    domain = DOMAIN_BY_MATRIX.get(matrix)
    if domain is None:
        return None

    versions: dict[str, str] = {
        "navigator": NAVIGATOR_VERSION,
        "layer": LAYER_FORMAT_VERSION,
    }
    if domain != "atlas-atlas":
        versions = {"attack": ATTACK_VERSION, **versions}

    layer_techniques = []
    seen: set[str] = set()
    scored: set[str] = set()
    # Parents of any scored sub-technique, so we can expand them below —
    # Navigator collapses sub-techniques by default, hiding a scored one until
    # its parent is expanded. Maps parent ID -> the child's tactic.
    parents_to_expand: dict[str, str | None] = {}
    for technique_id, tactic in techniques:
        if not technique_id or technique_id in seen:
            continue
        seen.add(technique_id)
        scored.add(technique_id)
        entry: dict[str, object] = {
            "techniqueID": technique_id,
            "score": 1,
            "color": HIGHLIGHT_COLOR,
            "enabled": True,
        }
        if tactic:
            entry["tactic"] = tactic
        if "." in technique_id:
            parent_id = technique_id.split(".", 1)[0]
            parents_to_expand.setdefault(parent_id, tactic)
        layer_techniques.append(entry)

    # Expand each parent so its scored sub-technique is visible. If the parent
    # is itself scored, flag its existing entry; otherwise add a bare
    # expand-only entry (no highlight), mirroring MITRE's own ATLAS layers.
    for parent_id, tactic in parents_to_expand.items():
        if parent_id in scored:
            for entry in layer_techniques:
                if entry["techniqueID"] == parent_id:
                    entry["showSubtechniques"] = True
            continue
        parent_entry: dict[str, object] = {
            "techniqueID": parent_id,
            "showSubtechniques": True,
        }
        if tactic:
            parent_entry["tactic"] = tactic
        layer_techniques.append(parent_entry)

    return {
        "name": name,
        "versions": versions,
        "domain": domain,
        "description": description,
        "techniques": layer_techniques,
        "legendItems": [{"label": LEGEND_LABEL, "color": HIGHLIGHT_COLOR}],
        # Show technique IDs in every cell and lay sub-techniques out to the
        # side so the scenario's picks are easy to read off the matrix.
        "layout": {
            "layout": "side",
            "showID": True,
            "showName": True,
        },
    }


def navigator_for_domain(domain: str) -> tuple[str, str]:
    """Return the ``(name, url)`` of the Navigator that loads ``domain`` layers.

    ATLAS layers only load in the ATLAS Navigator fork; everything else is an
    ATT&CK domain served by the MITRE ATT&CK Navigator.
    """
    if domain == DOMAIN_BY_MATRIX["ATLAS"]:
        return "ATLAS Navigator", ATLAS_NAVIGATOR_URL
    return "ATT&CK Navigator", ATTACK_NAVIGATOR_URL


def dumps(layer: dict) -> str:
    """Serialise a layer dict to indented JSON for download."""
    return json.dumps(layer, indent=2)


def layer_filename(base: str) -> str:
    """Derive the layer's download filename from the scenario's.

    ``"threat_group_scenario.md"`` -> ``"threat_group_scenario_layer.json"``.
    """
    stem = base[:-3] if base.endswith(".md") else base
    return f"{stem}_layer.json"
