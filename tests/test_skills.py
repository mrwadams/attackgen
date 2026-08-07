"""Tests for the bundled Agent Skills under `skills/`.

A skill is prose, so there is nothing to unit-test in the usual sense. What can
rot silently is its *references*: it names MCP tools, reference files and
framework vocabulary, and none of those are imports, so renaming a tool or a
threat category would leave the skill quietly instructing a client to call
something that no longer exists. These tests pin those cross-references.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

import data.ai_insider_threats as aip
import mcp_server

SKILL_DIR = Path(__file__).resolve().parent.parent / "skills" / "attackgen-tabletop"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES = sorted((SKILL_DIR / "references").glob("*.md"))


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def all_text(skill_text: str) -> str:
    """SKILL.md plus every reference — the whole instruction surface."""
    return skill_text + "\n".join(p.read_text(encoding="utf-8") for p in REFERENCES)


@pytest.fixture(scope="module")
def registered_tools() -> set[str]:
    return {tool.name for tool in asyncio.run(mcp_server.mcp.list_tools())}


class TestFrontmatter:
    def test_has_the_required_keys(self, skill_text: str) -> None:
        assert skill_text.startswith("---\n")
        frontmatter = skill_text.split("---", 2)[1]
        for key in ("name:", "description:", "version:", "license:"):
            assert key in frontmatter

    def test_name_matches_the_directory(self, skill_text: str) -> None:
        assert f"name: {SKILL_DIR.name}" in skill_text

    def test_description_covers_both_paths(self, skill_text: str) -> None:
        """The description is the only thing a client sees when deciding whether
        to load the skill, so it has to mention the AI-insider path or that path
        is unreachable in practice."""
        description = skill_text.split("---", 2)[1].split("description:")[1]
        description = description.split("version:")[0].lower()
        assert "att&ck" in description
        assert "insider" in description


class TestCrossReferences:
    def test_referenced_tools_exist(self, all_text: str, registered_tools: set[str]) -> None:
        """Every `some_tool(...)` mention must be a real registered MCP tool."""
        mentioned = set(re.findall(r"`([a-z_]+)\(", all_text))
        unknown = mentioned - registered_tools
        assert not unknown, f"skill references non-existent MCP tools: {sorted(unknown)}"

    def test_both_paths_name_their_entry_tools(self, skill_text: str) -> None:
        for tool in ("get_kill_chain", "get_detection_report",
                     "list_ai_insider_options", "get_ai_insider_prompt"):
            assert tool in skill_text, tool

    @pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.name)
    def test_reference_files_are_linked_from_somewhere(
        self, path: Path, all_text: str
    ) -> None:
        """An unreferenced file under `references/` is never loaded, so it is
        dead weight the client pays for in the folder listing alone."""
        others = "".join(
            p.read_text(encoding="utf-8") for p in [SKILL_MD, *REFERENCES] if p != path
        )
        assert f"references/{path.name}" in others

    def test_linked_reference_files_exist(self, all_text: str) -> None:
        for name in set(re.findall(r"references/([\w.-]+\.md)", all_text)):
            assert (SKILL_DIR / "references" / name).is_file(), name

    def test_html_template_exists(self, all_text: str) -> None:
        assert "assets/tabletop-report.html" in all_text
        assert (SKILL_DIR / "assets" / "tabletop-report.html").is_file()


class TestAiInsiderVocabulary:
    """The Path B reference tells the client to render specific framework
    concepts. If the data module renames one, the instruction goes stale."""

    def test_archetype_naming_convention_holds(self, all_text: str) -> None:
        """`ai-insider-format.md` quotes an archetype verbatim as an example and
        describes the levels as L1-L4."""
        assert "Human-as-Auditor (L4 — Critical Threat)" in all_text
        assert "Human-as-Auditor (L4 — Critical Threat)" in aip.DEPLOYMENT_ARCHETYPES

    def test_archetype_fields_the_reference_asks_for_exist(self) -> None:
        """The reference asks for an archetype block of access, detection
        posture, primary threats and critical control."""
        for archetype in aip.DEPLOYMENT_ARCHETYPES.values():
            for field in ("access", "detection", "primary_threats", "critical_control"):
                assert archetype.get(field), field

    def test_required_decision_leads_are_quoted_accurately(self, all_text: str) -> None:
        """The reference names four decision leads as its example. Each must
        still be the lead of a real decision on some template."""
        leads = {
            decision.split(" — ", 1)[0]
            for name in aip.AI_INSIDER_TEMPLATES
            for decision in aip.resolve_template(name)["required_decisions"]
        }
        for quoted in ("Containment", "Identity rotation",
                       "Third-party notification", "Evidence preservation"):
            assert f"*{quoted}*" in all_text, quoted
            assert quoted in leads, quoted
