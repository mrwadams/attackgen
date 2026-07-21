"""Headless MITRE ATT&CK / ATLAS loaders + kill-chain resolution.

This is the framework-agnostic home for the data work that used to live inline
in the scenario pages: loading the STIX bundles, listing threat groups / case
studies, and turning a group alias (or ATLAS case study) into the sampled kill
chain that feeds the prompt. Both the Streamlit pages and the MCP server import
from here, so there is one implementation of the technique-selection logic.

No Streamlit. Data paths are resolved from this file's location (not the cwd),
so the MCP server can run from anywhere. The heavy ``MitreAttackData`` bundles
(enterprise-attack.json is ~53 MB) are loaded lazily behind ``lru_cache``
singletons — the first data call pays the cost, later calls are free, and merely
importing this module (e.g. at MCP-server startup) loads nothing.

Parity with the old page-1 logic is deliberate and load-bearing:
  * Enterprise/ICS sample **one technique per kill-chain phase**, drawn from the
    *non-deduplicated* technique set (so a technique listed under a phase several
    times is proportionally more likely to be picked) — matching the UI, whose
    kill chain varies run to run. ``resolve_threat_group_kill_chain`` is a plain
    function over cached data (not itself cached), so each call resamples; pass
    ``seed`` for a deterministic draw.
  * ATLAS uses the case study's full documented procedure, no sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd
from mitreattack.stix20 import MitreAttackData

from atlas_parser import ATLASData, get_techniques_from_case_study_procedure

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Kill-chain phase order, lifted verbatim from page 1. Techniques are sorted into
# this order before the kill chain is assembled.
PHASE_ORDER_ATTACK = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution", "Persistence",
    "Privilege Escalation", "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
    "Collection", "Command and Control", "Exfiltration", "Impact",
]
PHASE_ORDER_ATLAS = [
    "Reconnaissance", "Resource Development", "Initial Access", "AI Model Access",
    "Execution", "Persistence", "Privilege Escalation", "Defense Evasion",
    "Credential Access", "Discovery", "Lateral Movement", "Collection",
    "AI Attack Staging", "Command and Control", "Exfiltration", "Impact",
]


# ---------------------------------------------------------------------------
# Lazy, cached bundle loaders
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def enterprise_data() -> MitreAttackData:
    return MitreAttackData(str(_DATA_DIR / "enterprise-attack.json"))


@lru_cache(maxsize=1)
def ics_data() -> MitreAttackData:
    return MitreAttackData(str(_DATA_DIR / "ics-attack.json"))


@lru_cache(maxsize=1)
def atlas_data() -> ATLASData:
    return ATLASData(str(_DATA_DIR / "stix-atlas.json"))


def mitre_data_for_matrix(matrix: str) -> MitreAttackData:
    """Return the ``MitreAttackData`` bundle for an ATT&CK matrix.

    Only ``"Enterprise"`` and ``"ICS"`` have ``MitreAttackData`` bundles; ATLAS
    uses ``atlas_data()`` instead. Raises ``ValueError`` for anything else.
    """
    if matrix == "Enterprise":
        return enterprise_data()
    if matrix == "ICS":
        return ics_data()
    raise ValueError(f"No MitreAttackData bundle for matrix '{matrix}' (use atlas_data() for ATLAS).")


def load_attack_data() -> dict:
    """The ``{"enterprise","ics","atlas"}`` bundle dict the pages expect.

    Kept for the pages, which index it by ``matrix.lower()``. Backed by the same
    cached singletons, so importing/using it never loads a bundle twice.
    """
    return {"enterprise": enterprise_data(), "ics": ics_data(), "atlas": atlas_data()}


# ---------------------------------------------------------------------------
# Group / case-study / technique listings
# ---------------------------------------------------------------------------

_GROUPS_FILE = {"Enterprise": "groups.json", "ICS": "groups_ics.json"}
_CAMPAIGNS_FILE = {"Enterprise": "campaigns.json", "ICS": "campaigns_ics.json"}


@lru_cache(maxsize=4)
def list_threat_groups(matrix: str) -> tuple[dict, ...]:
    """Threat groups (Enterprise/ICS) or case studies (ATLAS) as ``{group,url}``.

    Returns a tuple (hashable, so the ``lru_cache`` is happy) of JSON-native
    dicts. For ATLAS this returns the case studies keyed the same way.
    """
    if matrix == "ATLAS":
        return tuple({"group": c["group"], "url": c["url"]} for c in list_case_studies())
    filename = _GROUPS_FILE.get(matrix)
    if filename is None:
        raise ValueError(f"Unknown matrix '{matrix}'.")
    df = pd.read_json(str(_DATA_DIR / filename))
    return tuple({"group": str(r["group"]), "url": str(r["url"])} for _, r in df.iterrows())


@lru_cache(maxsize=4)
def list_campaigns(matrix: str) -> tuple[dict, ...]:
    """Documented ATT&CK campaigns (Enterprise/ICS) as ``{group,url}`` dicts.

    Mirrors ``list_threat_groups``; the ``"group"`` key is reused so the page and
    MCP listing code stay uniform. Campaigns exist only in the Enterprise and ICS
    matrices — ATLAS has no campaign objects and raises ``ValueError``.
    """
    filename = _CAMPAIGNS_FILE.get(matrix)
    if filename is None:
        raise ValueError(f"No campaigns for matrix '{matrix}' (Enterprise/ICS only).")
    df = pd.read_json(str(_DATA_DIR / filename))
    return tuple({"group": str(r["group"]), "url": str(r["url"])} for _, r in df.iterrows())


@lru_cache(maxsize=1)
def list_case_studies() -> tuple[dict, ...]:
    """ATLAS case studies with a short summary, as JSON-native dicts."""
    df = pd.read_json(str(_DATA_DIR / "atlas-case-studies.json"))
    out = []
    for _, r in df.iterrows():
        out.append({
            "group": str(r["group"]),
            "url": str(r["url"]),
            "summary": str(r.get("summary", "")) if r.get("summary") is not None else "",
            "case_study_type": str(r.get("case_study_type", "")) if r.get("case_study_type") is not None else "",
        })
    return tuple(out)


def list_technique_options(matrix: str) -> list[dict]:
    """All selectable techniques for a matrix, for the Custom page multiselect.

    Each entry: ``{"id","Technique Name","External ID","Display Name"}`` where
    ``Display Name`` is ``"Name (ID)"``. Mirrors the page-2 ``load_techniques``.
    """
    if matrix == "ATLAS":
        techniques = atlas_data().get_techniques()
        return [
            {
                "id": tech["id"],
                "Technique Name": tech["name"],
                "External ID": tech["external_id"],
                "Display Name": f"{tech['name']} ({tech['external_id']})",
            }
            for tech in techniques
        ]

    techniques = mitre_data_for_matrix(matrix).get_techniques()
    rows: list[dict] = []
    for technique in techniques:
        for reference in technique.external_references:
            if "external_id" in reference:
                rows.append({
                    "id": technique.id,
                    "Technique Name": technique.name,
                    "External ID": reference["external_id"],
                    "Display Name": f"{technique.name} ({reference['external_id']})",
                })
    return rows


# ---------------------------------------------------------------------------
# Kill-chain resolution
# ---------------------------------------------------------------------------


@dataclass
class KillChain:
    """A resolved kill chain for one threat group / case study.

    ``techniques`` is the set actually fed to the model (sampled one-per-phase for
    Enterprise/ICS; the full procedure for ATLAS). ``all_techniques`` is the full
    deduplicated set for display. Each is a list of JSON-native dicts with keys
    ``Technique Name`` / ``ATT&CK ID`` / ``Phase Name``. ``kill_chain_string`` is
    the ``"Phase: Name (ID)"`` newline-joined form the prompt consumes.
    """

    matrix: str
    group_alias: str
    techniques: list[dict]
    kill_chain_string: str
    all_techniques: list[dict] = field(default_factory=list)


def _kill_chain_string(techniques: list[dict]) -> str:
    return "\n".join(
        f"{t['Phase Name']}: {t['Technique Name']} ({t['ATT&CK ID']})" for t in techniques
    )


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-native records (str-cast so no numpy/Categorical leaks)."""
    return [
        {
            "Technique Name": str(row["Technique Name"]),
            "ATT&CK ID": str(row["ATT&CK ID"]),
            "Phase Name": str(row["Phase Name"]),
        }
        for _, row in df.iterrows()
    ]


def resolve_threat_group_kill_chain(
    matrix: str, group_alias: str, *, seed: int | None = None
) -> KillChain:
    """Resolve an Enterprise/ICS threat group to a sampled kill chain.

    Replicates the page-1 logic exactly: resolve the group's techniques, derive
    name/ID/phase, dedupe for display, normalise phase names, order by
    ``PHASE_ORDER_ATTACK``, then sample one technique per phase from the
    *non-deduplicated* set. ``seed`` makes the sampling deterministic; ``None``
    reproduces the UI's per-call randomness. Returns a ``KillChain`` with empty
    technique lists when the group has no associated techniques.
    """
    mitre = mitre_data_for_matrix(matrix)
    group = mitre.get_groups_by_alias(group_alias)
    if not group:
        return KillChain(matrix, group_alias, [], "", [])

    techniques = mitre.get_techniques_used_by_group(group[0].id)
    return _kill_chain_from_relationships(
        matrix, group_alias, techniques, mitre, sample=True, seed=seed
    )


def resolve_campaign_kill_chain(matrix: str, campaign_alias: str) -> KillChain:
    """Resolve an Enterprise/ICS campaign to its full documented kill chain.

    A campaign is a documented real-world intrusion, so — unlike a threat group —
    every technique observed in the campaign is replayed (no per-phase sampling),
    matching ``resolve_case_study_kill_chain``'s behaviour for ATLAS. Techniques
    are still deduped for display, phase-normalised and ordered by
    ``PHASE_ORDER_ATTACK``. Returns an empty ``KillChain`` when the campaign is
    unknown or has no associated techniques.
    """
    mitre = mitre_data_for_matrix(matrix)
    campaign = mitre.get_campaigns_by_alias(campaign_alias)
    if not campaign:
        return KillChain(matrix, campaign_alias, [], "", [])

    techniques = mitre.get_techniques_used_by_campaign(campaign[0].id)
    return _kill_chain_from_relationships(
        matrix, campaign_alias, techniques, mitre, sample=False
    )


def _kill_chain_from_relationships(
    matrix: str,
    alias: str,
    techniques: list,
    mitre: MitreAttackData,
    *,
    sample: bool,
    seed: int | None = None,
) -> KillChain:
    """Turn ATT&CK ``uses`` relationships into a ``KillChain``.

    Shared by the group and campaign resolvers. ``techniques`` is the relationship
    list returned by ``get_techniques_used_by_group`` /
    ``get_techniques_used_by_campaign`` (each entry has an ``object`` technique).
    Derives name/ID/phase, dedupes for display, normalises phase names and orders
    by ``PHASE_ORDER_ATTACK``. When ``sample`` is true, one technique per phase is
    drawn from the *non-deduplicated* set (threat-group behaviour; ``seed`` makes
    the draw deterministic); when false, the full deduped set is used (campaign
    behaviour — the whole documented intrusion).
    """
    if not techniques:
        return KillChain(matrix, alias, [], "", [])

    techniques_df = pd.DataFrame(techniques)
    techniques_df_llm = techniques_df.copy()
    techniques_df["Technique Name"] = techniques_df_llm["Technique Name"] = techniques_df["object"].apply(lambda x: x["name"])
    techniques_df["ATT&CK ID"] = techniques_df_llm["ATT&CK ID"] = techniques_df["object"].apply(lambda x: mitre.get_attack_id(x["id"]))
    techniques_df["Phase Name"] = techniques_df_llm["Phase Name"] = techniques_df["object"].apply(lambda x: x["kill_chain_phases"][0]["phase_name"])
    techniques_df = techniques_df.drop_duplicates(["Phase Name", "Technique Name", "ATT&CK ID"])

    for df in (techniques_df, techniques_df_llm):
        df["Phase Name"] = df["Phase Name"].str.replace("-", " ").str.title()
        df["Phase Name"] = df["Phase Name"].replace("Command And Control", "Command and Control")

    phase_dtype = pd.CategoricalDtype(categories=PHASE_ORDER_ATTACK, ordered=True)
    techniques_df["Phase Name"] = techniques_df["Phase Name"].astype(phase_dtype)
    techniques_df_llm["Phase Name"] = techniques_df_llm["Phase Name"].astype(phase_dtype)
    techniques_df = techniques_df.sort_values("Phase Name")
    techniques_df_llm = techniques_df_llm.sort_values("Phase Name")

    all_records = _records(techniques_df.sort_values("Phase Name"))

    if not sample:
        # Full documented chain (campaign / no sampling): techniques == all.
        return KillChain(
            matrix=matrix,
            group_alias=alias,
            techniques=all_records,
            kill_chain_string=_kill_chain_string(all_records),
            all_techniques=all_records,
        )

    selected_techniques_df = (
        techniques_df_llm.groupby("Phase Name", observed=False)
        .apply(
            lambda x: x.sample(n=1, random_state=seed) if not x.empty else pd.DataFrame(columns=x.columns),
            include_groups=False,
        )
        .reset_index()
    )

    sampled_records = _records(selected_techniques_df)
    return KillChain(
        matrix=matrix,
        group_alias=alias,
        techniques=sampled_records,
        kill_chain_string=_kill_chain_string(sampled_records),
        all_techniques=all_records,
    )


def resolve_case_study_kill_chain(case_study_name: str) -> KillChain:
    """Resolve an ATLAS case study to its full documented kill chain (no sampling).

    Reads the case study's ``procedure`` from ``atlas-case-studies.json``, walks
    it into techniques via ``get_techniques_from_case_study_procedure``, and
    orders them by ``PHASE_ORDER_ATLAS`` — matching page 1's ATLAS branch.
    """
    atlas = atlas_data()
    df = pd.read_json(str(_DATA_DIR / "atlas-case-studies.json"))
    match = df[df["group"] == case_study_name]
    if match.empty:
        return KillChain("ATLAS", case_study_name, [], "", [])

    procedure = match.iloc[0].get("procedure", [])
    if not isinstance(procedure, list) or not procedure:
        return KillChain("ATLAS", case_study_name, [], "", [])

    techniques_list = get_techniques_from_case_study_procedure(procedure, atlas)
    if not techniques_list:
        return KillChain("ATLAS", case_study_name, [], "", [])

    techniques_df = pd.DataFrame(techniques_list)
    phase_dtype = pd.CategoricalDtype(categories=PHASE_ORDER_ATLAS, ordered=True)
    techniques_df["Phase Name"] = techniques_df["Phase Name"].astype(phase_dtype)
    techniques_df = techniques_df.sort_values("Phase Name")

    records = _records(techniques_df)
    return KillChain(
        matrix="ATLAS",
        group_alias=case_study_name,
        techniques=records,
        kill_chain_string=_kill_chain_string(records),
        all_techniques=records,
    )
