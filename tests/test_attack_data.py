"""Tests for `core.attack_data` — headless loaders + kill-chain resolution.

Enterprise/ICS resolution is exercised with a fake `MitreAttackData` covering the
three helpers used, so the 53 MB bundle is never loaded. ATLAS resolution runs
against the real bundled data. The tests pin the parity-critical behaviours:
phase normalisation, dedupe for display, one-technique-per-phase sampling, the
kill-chain string format, seed determinism, and JSON-native output.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import core.attack_data as ad


class FakeMitreData:
    """Stand-in for `MitreAttackData` covering the three resolver helpers.

    Models a group with two phases; "Command And Control" is title-cased and must
    normalise to "Command and Control". The Initial Access phase has two distinct
    techniques so one-per-phase sampling has a real choice to make.
    """

    _TECHS = [
        ("attack-pattern--1", "Spearphishing Attachment", "T1566.001", "initial-access"),
        ("attack-pattern--2", "Drive-by Compromise", "T1189", "initial-access"),
        ("attack-pattern--3", "Web Protocols", "T1071.001", "command-and-control"),
    ]

    def get_groups_by_alias(self, alias):
        if alias == "TESTGROUP":
            return [SimpleNamespace(id="intrusion-set--x")]
        return []

    def get_techniques_used_by_group(self, stix_id):
        return [
            {"object": {"id": sid, "name": name, "kill_chain_phases": [{"phase_name": phase}]}}
            for sid, name, _ext, phase in self._TECHS
        ]

    def get_attack_id(self, stix_id):
        return {sid: ext for sid, _n, ext, _p in self._TECHS}[stix_id]


class FakeTechniqueData:
    """Stand-in for `MitreAttackData.get_techniques()` for list_technique_options.

    Each technique carries multiple external_references; only those with an
    ``external_id`` should produce a row, so the CAPEC ref below must be skipped.
    """

    def get_techniques(self):
        return [
            SimpleNamespace(
                id="attack-pattern--1",
                name="Spearphishing Attachment",
                external_references=[
                    {"source_name": "mitre-attack", "external_id": "T1566.001"},
                    {"source_name": "capec"},  # no external_id -> must be skipped
                ],
            ),
            SimpleNamespace(
                id="attack-pattern--2",
                name="Drive-by Compromise",
                external_references=[
                    {"source_name": "mitre-attack", "external_id": "T1189"},
                ],
            ),
        ]


@pytest.fixture
def fake_enterprise(monkeypatch):
    monkeypatch.setattr(ad, "mitre_data_for_matrix", lambda matrix: FakeMitreData())
    return FakeMitreData()


class TestResolveThreatGroup:
    def test_unknown_group_returns_empty(self, fake_enterprise):
        kc = ad.resolve_threat_group_kill_chain("Enterprise", "NOPE")
        assert kc.techniques == [] and kc.all_techniques == [] and kc.kill_chain_string == ""

    def test_phase_normalised_and_one_per_phase(self, fake_enterprise):
        kc = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=0)
        phases = [t["Phase Name"] for t in kc.techniques]
        # One sampled technique per phase, phases ordered per PHASE_ORDER_ATTACK.
        assert phases == ["Initial Access", "Command and Control"]
        # The hyphenated raw phase was title-cased + fixed.
        assert "Command and Control" in phases
        # all_techniques holds the full deduped display set (3 techniques).
        assert len(kc.all_techniques) == 3

    def test_kill_chain_string_format(self, fake_enterprise):
        kc = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=0)
        first = kc.kill_chain_string.splitlines()[0]
        # "Phase: Name (ID)"
        assert first.startswith("Initial Access: ")
        assert first.endswith(")") and "(" in first

    def test_seed_is_deterministic(self, fake_enterprise):
        a = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=7)
        b = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=7)
        assert a.kill_chain_string == b.kill_chain_string

    def test_output_is_json_native(self, fake_enterprise):
        kc = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=1)
        # Must not raise (no numpy / pandas Categorical leaking through).
        json.dumps(kc.techniques)
        json.dumps(kc.all_techniques)
        for t in kc.techniques:
            assert set(t) == {"Technique Name", "ATT&CK ID", "Phase Name"}
            assert all(isinstance(v, str) for v in t.values())


@pytest.mark.parametrize("seed", [0, 42])
def test_sampling_covers_every_phase_present(fake_enterprise, seed):
    kc = ad.resolve_threat_group_kill_chain("Enterprise", "TESTGROUP", seed=seed)
    # Two phases present in the fake -> exactly two sampled techniques.
    assert len(kc.techniques) == 2


class TestResolveCaseStudyAtlas:
    """ATLAS resolution against the real bundled data (deterministic, no sampling)."""

    def test_known_case_study_resolves(self):
        studies = ad.list_case_studies()
        assert studies, "expected bundled ATLAS case studies"
        name = studies[0]["group"]
        kc = ad.resolve_case_study_kill_chain(name)
        assert kc.matrix == "ATLAS"
        assert kc.techniques, "case study should yield techniques"
        # No sampling: sampled set == full set for ATLAS.
        assert kc.techniques == kc.all_techniques
        json.dumps(kc.techniques)

    def test_unknown_case_study_returns_empty(self):
        kc = ad.resolve_case_study_kill_chain("no-such-case-study-xyz")
        assert kc.techniques == [] and kc.kill_chain_string == ""


class TestListings:
    def test_list_threat_groups_enterprise_shape(self):
        groups = ad.list_threat_groups("Enterprise")
        assert groups and set(groups[0]) == {"group", "url"}

    def test_unknown_matrix_raises(self):
        with pytest.raises(ValueError):
            ad.list_threat_groups("Mobile")


class TestMitreDataForMatrix:
    def test_rejects_atlas_and_unknown(self):
        # ATLAS has no MitreAttackData bundle; anything unknown is a caller error.
        # Both raise before any (53 MB) bundle load.
        with pytest.raises(ValueError):
            ad.mitre_data_for_matrix("ATLAS")
        with pytest.raises(ValueError):
            ad.mitre_data_for_matrix("Mobile")


class TestListTechniqueOptions:
    def test_attack_branch_builds_display_names(self, monkeypatch):
        monkeypatch.setattr(ad, "mitre_data_for_matrix", lambda matrix: FakeTechniqueData())
        opts = ad.list_technique_options("Enterprise")
        # Three external_references across two techniques, but the CAPEC ref has
        # no external_id -> exactly two rows.
        assert len(opts) == 2
        first = opts[0]
        assert set(first) == {"id", "Technique Name", "External ID", "Display Name"}
        assert first["External ID"] == "T1566.001"
        assert first["Display Name"] == "Spearphishing Attachment (T1566.001)"

    def test_atlas_branch_against_real_bundle(self):
        opts = ad.list_technique_options("ATLAS")
        assert opts, "expected bundled ATLAS techniques"
        row = opts[0]
        assert set(row) == {"id", "Technique Name", "External ID", "Display Name"}
        assert row["Display Name"] == f"{row['Technique Name']} ({row['External ID']})"
