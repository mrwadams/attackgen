"""Tests for `core.custom_templates` — the Custom page's incident-response templates.

The guard tests run against the shipped STIX bundles on purpose: a data refresh
that renames, revokes or deprecates a template technique must fail here rather
than shrink a template on page 2 without a word. Loading the Enterprise bundle
costs a few seconds, once per session.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import core.attack_data as ad
import core.custom_templates as ct

_ENTRIES = [
    (matrix, template, label)
    for matrix, templates in ct.INCIDENT_RESPONSE_TEMPLATES.items()
    for template, labels in templates.items()
    for label in labels
]


@pytest.mark.parametrize(("matrix", "template", "label"), _ENTRIES)
def test_every_template_entry_is_an_active_technique(matrix, template, label):
    assert label in ad.active_technique_labels(matrix), (
        f"{matrix} template {template!r} names {label!r}, which is not an active "
        f"technique in the shipped bundle. Update the template after a data refresh."
    )


@pytest.mark.parametrize("matrix", ["Enterprise", "ICS", "ATLAS"])
def test_every_template_resolves_in_full(matrix):
    for template, labels in ct.INCIDENT_RESPONSE_TEMPLATES[matrix].items():
        kept, dropped = ct.resolve_incident_template(matrix, template)
        assert dropped == []
        assert kept == labels


def test_retired_entries_are_replaced():
    templates = ct.INCIDENT_RESPONSE_TEMPLATES
    phishing = templates["Enterprise"]["Phishing Attack"]
    assert "Spearphishing Attachment (T1566.001)" in phishing
    assert "Software Extensions (T1176)" in phishing
    assert "Registry Run Keys / Startup Folder (T1547.001)" in templates["Enterprise"]["Malware Infection"]
    assert "Command Message (T1691.001)" in templates["ICS"]["Denial of Service"]


class TestActiveTechniqueLabels:
    def test_excludes_revoked_enterprise_technique(self):
        labels = ad.active_technique_labels("Enterprise")
        assert "Spearphishing Attachment (T1193)" not in labels
        assert "Spearphishing Attachment (T1566.001)" in labels

    def test_excludes_revoked_ics_technique(self):
        labels = ad.active_technique_labels("ICS")
        assert "Block Command Message (T0803)" not in labels
        assert "Command Message (T1691.001)" in labels

    def test_atlas_uses_name_and_external_id(self):
        assert "LLM Prompt Injection (AML.T0051)" in ad.active_technique_labels("ATLAS")

    def test_unknown_matrix_raises(self):
        with pytest.raises(ValueError):
            ad.active_technique_labels("Mobile")


class TestResolveIncidentTemplate:
    def test_splits_kept_and_dropped_in_template_order(self, monkeypatch):
        monkeypatch.setitem(
            ct.INCIDENT_RESPONSE_TEMPLATES, "Enterprise", {"T": ["A (T1)", "B (T2)", "C (T3)"]}
        )
        monkeypatch.setattr(ad, "active_technique_labels", lambda matrix: frozenset({"A (T1)", "C (T3)"}))
        assert ct.resolve_incident_template("Enterprise", "T") == (["A (T1)", "C (T3)"], ["B (T2)"])

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError):
            ct.resolve_incident_template("Enterprise", "No Such Template")


def test_module_is_streamlit_free():
    tree = ast.parse(Path(ct.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "streamlit" not in imported
