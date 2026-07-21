"""Tests for `mcp_server` — tool registration, data tools, and generate tools.

Generate tools are exercised with the shared `mock_litellm_completion` fixture
and a monkeypatched kill-chain resolver, so no network or 53 MB bundle is
touched. LangSmith is forced off so `call_llm` takes the direct path.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import core.llm as llm_module
import mcp_server as s
from core.attack_data import KillChain
from data.ai_insider_threats import DEPLOYMENT_ARCHETYPES
from tests.test_detections import FakeMitreData


@pytest.fixture
def no_langsmith(monkeypatch):
    monkeypatch.setattr(llm_module, "_langsmith_client", None)


@pytest.fixture
def canned_kill_chain(monkeypatch):
    """Replace the resolvers so generate tools don't load real data."""
    kc = KillChain(
        matrix="Enterprise",
        group_alias="APT29",
        techniques=[
            {"Technique Name": "Phishing", "ATT&CK ID": "T1566", "Phase Name": "Initial Access"}
        ],
        kill_chain_string="Initial Access: Phishing (T1566)",
        all_techniques=[
            {"Technique Name": "Phishing", "ATT&CK ID": "T1566", "Phase Name": "Initial Access"}
        ],
    )
    monkeypatch.setattr(s.ad, "resolve_threat_group_kill_chain", lambda *a, **k: kc)
    monkeypatch.setattr(s.ad, "resolve_case_study_kill_chain", lambda *a, **k: kc)
    return kc


EXPECTED_TOOLS = {
    "list_threat_groups", "list_case_studies", "get_kill_chain", "get_detection_report",
    "get_navigator_layer", "list_ai_insider_options", "get_ai_insider_prompt",
    "generate_threat_group_scenario", "generate_custom_scenario", "generate_ai_insider_scenario",
}


def test_all_tools_registered_with_schemas():
    tools = asyncio.run(s.mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    for tool in tools:
        assert tool.inputSchema and tool.inputSchema.get("type") == "object"


class TestDataTools:
    def test_get_navigator_layer_enterprise(self):
        out = s.get_navigator_layer("Enterprise", ["T1059", "Phishing (T1566)"])
        assert out["layer_json"] is not None
        layer = json.loads(out["layer_json"])
        assert layer["domain"] == "enterprise-attack"

    def test_get_detection_report_uses_injected_data(self, monkeypatch):
        monkeypatch.setattr(s.ad, "mitre_data_for_matrix", lambda matrix: FakeMitreData())
        out = s.get_detection_report("Enterprise", ["T1059"])
        assert out["matrix"] == "Enterprise"
        assert out["markdown"] and "Analytic 1428" in out["markdown"]

    def test_list_ai_insider_options_keys(self):
        opts = s.list_ai_insider_options()
        assert set(opts) == {"archetypes", "threat_categories", "stride", "capabilities", "templates"}
        assert opts["archetypes"]

    def test_get_kill_chain_embeds_prompt_when_org_supplied(self, canned_kill_chain):
        out = s.get_kill_chain("Enterprise", "APT29", industry="Finance", company_size="Large")
        assert out["kill_chain_string"] == "Initial Access: Phishing (T1566)"
        assert out["messages"] is not None and out["messages"][0]["role"] == "system"

    def test_get_kill_chain_no_prompt_without_org(self, canned_kill_chain):
        out = s.get_kill_chain("Enterprise", "APT29")
        assert out["messages"] is None

    def test_get_ai_insider_prompt_builds_messages(self):
        archetype = next(iter(DEPLOYMENT_ARCHETYPES))
        out = s.get_ai_insider_prompt(
            archetype, categories=[], stride=[], capabilities=[],
            industry="Healthcare", company_size="Large",
        )
        msgs = out["messages"]
        assert msgs[0]["role"] == "system"
        assert archetype in msgs[1]["content"]


class TestGenerateTools:
    def test_generate_threat_group_scenario(self, no_langsmith, canned_kill_chain, mock_litellm_completion):
        mock_litellm_completion.content = "# Scenario\n\nBody."
        out = s.generate_threat_group_scenario(
            "Enterprise", "APT29", "Finance", "Large",
            provider="Anthropic API", model="claude-sonnet-5", api_key="k",
        )
        assert out == "# Scenario\n\nBody."
        _args, kwargs = mock_litellm_completion.calls[0]
        assert kwargs["model"] == "anthropic/claude-sonnet-5"
        assert kwargs["api_key"] == "k"

    def test_generate_custom_scenario(self, no_langsmith, mock_litellm_completion):
        mock_litellm_completion.content = "custom scenario"
        out = s.generate_custom_scenario(
            "Enterprise", ["Phishing (T1566)", "T1059"], "Finance", "Large",
            provider="OpenAI API", model="gpt-5.5", api_key="k",
        )
        assert out == "custom scenario"
        _args, kwargs = mock_litellm_completion.calls[0]
        assert "T1566" in kwargs["messages"][1]["content"]

    def test_bad_provider_raises(self, canned_kill_chain):
        with pytest.raises(ValueError):
            s.generate_threat_group_scenario(
                "Enterprise", "APT29", "F", "L", provider="Bogus", model="x"
            )

    def test_bad_model_raises(self, canned_kill_chain):
        with pytest.raises(ValueError):
            s.generate_threat_group_scenario(
                "Enterprise", "APT29", "F", "L", provider="Anthropic API", model="not-a-model"
            )

    def test_custom_provider_skips_model_validation(self, no_langsmith, mock_litellm_completion):
        mock_litellm_completion.content = "ok"
        out = s.generate_custom_scenario(
            "Enterprise", ["T1059"], "F", "L",
            provider="Custom", model="my-local-model", api_base="http://127.0.0.1:1234/v1",
        )
        assert out == "ok"

    def test_missing_model_raises(self, canned_kill_chain):
        with pytest.raises(ValueError):
            s.generate_threat_group_scenario(
                "Enterprise", "APT29", "F", "L", provider="Anthropic API", model=""
            )

    def test_generate_ai_insider_scenario(self, no_langsmith, mock_litellm_completion):
        archetype = next(iter(DEPLOYMENT_ARCHETYPES))
        mock_litellm_completion.content = "# AI Insider Scenario"
        out = s.generate_ai_insider_scenario(
            archetype, [], [], [], "Healthcare", "Large",
            provider="Anthropic API", model="claude-sonnet-5", api_key="k",
        )
        assert out == "# AI Insider Scenario"
        _args, kwargs = mock_litellm_completion.calls[0]
        assert kwargs["model"] == "anthropic/claude-sonnet-5"
        # The chosen archetype made it into the built prompt.
        assert archetype in kwargs["messages"][1]["content"]

    def test_empty_kill_chain_raises(self, monkeypatch):
        """The guard fires before the model is ever called (no LLM mock needed)."""
        empty = KillChain("Enterprise", "APT29", [], "", [])
        monkeypatch.setattr(s.ad, "resolve_threat_group_kill_chain", lambda *a, **k: empty)
        with pytest.raises(ValueError):
            s.generate_threat_group_scenario(
                "Enterprise", "APT29", "Finance", "Large",
                provider="Anthropic API", model="claude-sonnet-5", api_key="k",
            )

    def test_empty_techniques_raises(self):
        with pytest.raises(ValueError):
            s.generate_custom_scenario(
                "Enterprise", [], "Finance", "Large",
                provider="Anthropic API", model="claude-sonnet-5", api_key="k",
            )

    def test_threat_group_include_detection_appends_report(
        self, no_langsmith, mock_litellm_completion, monkeypatch
    ):
        # A T1059 kill chain so FakeMitreData resolves a detection join to append.
        kc = KillChain(
            matrix="Enterprise", group_alias="APT29",
            techniques=[{
                "Technique Name": "Command and Scripting Interpreter",
                "ATT&CK ID": "T1059", "Phase Name": "Execution",
            }],
            kill_chain_string="Execution: Command and Scripting Interpreter (T1059)",
            all_techniques=[],
        )
        monkeypatch.setattr(s.ad, "resolve_threat_group_kill_chain", lambda *a, **k: kc)
        monkeypatch.setattr(s.ad, "mitre_data_for_matrix", lambda matrix: FakeMitreData())
        mock_litellm_completion.content = "# Scenario body"
        out = s.generate_threat_group_scenario(
            "Enterprise", "APT29", "Finance", "Large",
            provider="Anthropic API", model="claude-sonnet-5", api_key="k",
            include_detection=True,
        )
        assert "# Scenario body" in out
        assert "\n\n---\n\n" in out  # detection section separator
        assert "Analytic 1428" in out  # the joined Detection & Response content

    def test_custom_include_detection_appends_report(
        self, no_langsmith, mock_litellm_completion, monkeypatch
    ):
        monkeypatch.setattr(s.ad, "mitre_data_for_matrix", lambda matrix: FakeMitreData())
        mock_litellm_completion.content = "custom body"
        out = s.generate_custom_scenario(
            "Enterprise", ["Command and Scripting Interpreter (T1059)"], "Finance", "Large",
            provider="OpenAI API", model="gpt-5.5", api_key="k",
            include_detection=True,
        )
        assert "custom body" in out
        assert "\n\n---\n\n" in out
        assert "Analytic 1428" in out
