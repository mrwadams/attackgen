"""Compact views over a long scenario and over the inputs that produced it.

Two pure surfaces, both Streamlit-free so they can be tested (and reused)
without rendering anything:

* :func:`summarise_scenario` turns generated Markdown into an overview, a
  section index and a handful of facilitator shortcuts (injects, success
  criteria, metrics, artefacts, rules of engagement). A finished exercise runs
  to a dozen-plus sections, so the coordinator renders this above the result
  rather than truncating or rewriting the Markdown itself.
* :func:`describe_inputs` / :func:`summary_line` render the Generate-time input
  snapshot as human-readable facts. The same helpers describe what *will* be
  generated before the click and what *did* produce the result afterwards, so
  the two can never disagree about what a field means.

:func:`inputs_changed` answers "does the result on screen still match the form?"
It deliberately ignores the fields a page re-derives on every rerun — page 1
resamples one technique per phase each time it runs, so comparing those would
report every rerun as an edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

_WORDS_PER_MINUTE = 220
"""Reading pace used for the result's "~n min read" estimate."""


@dataclass(frozen=True)
class Section:
    """One Markdown heading, with the anchor Streamlit gives it."""

    level: int
    title: str
    anchor: str


@dataclass(frozen=True)
class ScenarioSummary:
    """A compact, navigable view of one generated scenario."""

    title: str | None = None
    overview: str = ""
    sections: tuple[Section, ...] = ()
    quick_links: tuple[tuple[str, Section], ...] = ()
    word_count: int = 0
    reading_minutes: int = 0

    @property
    def is_useful(self) -> bool:
        """Is there enough structure here for the navigation surface to help?"""
        return bool(self.sections) or bool(self.overview)


# Facilitator shortcuts, in the order an exercise is actually run. Each entry is
# (label, heading keywords); the first heading matching any keyword wins.
_QUICK_LINK_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Injects", ("inject", "timeline of events")),
    ("Discussion questions", ("discussion question",)),
    ("Success criteria", ("success criteria", "success measure", "objectives")),
    ("Metrics", ("metric", "measurement")),
    ("Artefacts", ("artefact", "artifact")),
    (
        "Rules of engagement",
        ("rules of engagement", "safety", "ground rules", "guardrails"),
    ),
)


def slugify(text: str) -> str:
    """Approximate the anchor Streamlit generates for a Markdown heading."""
    slug = re.sub(r"[^\w\- ]+", "", text.strip().lower())
    slug = re.sub(r"[\s_]+", "-", slug)
    return re.sub(r"-{2,}", "-", slug).strip("-")


def _strip_inline_markup(text: str) -> str:
    """Drop the emphasis/link syntax so a heading reads as plain text."""
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.strip()


def extract_sections(markdown: str) -> tuple[Section, ...]:
    """Every heading in ``markdown``, ignoring anything inside a code fence."""
    sections: list[Section] = []
    in_fence = False
    for line in (markdown or "").splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(line)
        if not match:
            continue
        title = _strip_inline_markup(match.group(2))
        if not title:
            continue
        sections.append(
            Section(level=len(match.group(1)), title=title, anchor=slugify(title))
        )
    return tuple(sections)


def _overview(markdown: str) -> str:
    """The first prose paragraph, as a one-glance summary of the scenario."""
    in_fence = False
    paragraph: list[str] = []
    for line in (markdown or "").splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.strip()
        if not stripped or _HEADING.match(line):
            if paragraph:
                break
            continue
        if stripped.startswith(("-", "*", "|", ">")) and not paragraph:
            continue
        paragraph.append(stripped)
    text = _strip_inline_markup(" ".join(paragraph))
    if len(text) > 400:
        text = text[:397].rsplit(" ", 1)[0] + "…"
    return text


def _quick_links(sections: tuple[Section, ...]) -> tuple[tuple[str, Section], ...]:
    links: list[tuple[str, Section]] = []
    for label, keywords in _QUICK_LINK_KEYWORDS:
        for section in sections:
            lowered = section.title.lower()
            if any(keyword in lowered for keyword in keywords):
                links.append((label, section))
                break
    return tuple(links)


def summarise_scenario(markdown: str) -> ScenarioSummary:
    """Summarise a generated scenario for the result's navigation surface."""
    sections = extract_sections(markdown)
    words = len(_WORD.findall(markdown or ""))
    title = next((s.title for s in sections if s.level == 1), None)
    return ScenarioSummary(
        title=title,
        overview=_overview(markdown),
        sections=sections,
        quick_links=_quick_links(sections),
        word_count=words,
        reading_minutes=max(1, round(words / _WORDS_PER_MINUTE)) if words else 0,
    )


# --- The Generate-time input snapshot ----------------------------------------

_ENTITY_LABELS = {
    "threat actor group": "Threat group",
    "case study": "Case study",
    "template": "Template",
    "deployment_archetype": "Deployment archetype",
}

_MODIFIER_LABELS = {
    "ai_uplift": "AI-enhanced adversary",
    "purple_team_narrative": "Purple-team narrative",
}

_MATRIX_LABELS = {
    "Enterprise": "Enterprise ATT&CK",
    "ICS": "ICS ATT&CK",
    "ATLAS": "MITRE ATLAS",
}


def modifier_labels(snapshot: dict) -> tuple[str, ...]:
    """Human labels for the modifiers that were on when Generate was pressed."""
    modifiers = snapshot.get("modifiers") or {}
    labels: list[str] = []
    for key, value in modifiers.items():
        if not value:
            continue
        if key in _MODIFIER_LABELS:
            labels.append(_MODIFIER_LABELS[key])
        elif isinstance(value, str):
            labels.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return tuple(labels)


def _entity(snapshot: dict) -> tuple[str, str] | None:
    entity = snapshot.get("selected_entity") or None
    if not entity or not entity.get("name"):
        return None
    label = _ENTITY_LABELS.get(str(entity.get("type", "")).lower(), "Selection")
    return label, str(entity["name"])


def _format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M UTC" if parsed.tzinfo else "%Y-%m-%d %H:%M")


def describe_inputs(snapshot: dict | None) -> tuple[tuple[str, str], ...]:
    """Describe a snapshot as ordered ``(label, value)`` facts.

    Used both before generation ("this is what will be sent") and with a
    finished result ("this is what produced it"), so the two always agree.
    """
    if not snapshot:
        return ()
    facts: list[tuple[str, str]] = []

    matrix = snapshot.get("matrix")
    if matrix:
        facts.append(("Framework", _MATRIX_LABELS.get(matrix, str(matrix))))

    organisation = snapshot.get("organisation") or {}
    if organisation.get("industry"):
        facts.append(("Industry", str(organisation["industry"])))
    if organisation.get("company_size"):
        facts.append(("Company size", str(organisation["company_size"])))

    entity = _entity(snapshot)
    if entity:
        facts.append(entity)

    selected = snapshot.get("selected_techniques") or []
    sampled = snapshot.get("sampled_techniques") or []
    if selected or sampled:
        count = len(sampled) if sampled else len(selected)
        noun = "technique" if count == 1 else "techniques"
        facts.append(("Techniques", f"{count} {noun}"))

    for key, label in (
        ("selected_categories", "Threat categories"),
        ("selected_stride", "STRIDE threats"),
        ("selected_capabilities", "Agent capabilities"),
    ):
        values = snapshot.get(key) or []
        if values:
            facts.append((label, ", ".join(str(v) for v in values)))

    if snapshot.get("scenario_seed"):
        facts.append(("Scenario seed", "Provided"))
    if snapshot.get("required_decisions"):
        facts.append(
            (
                "Required decisions",
                "; ".join(
                    str(d).split(" — ", 1)[0] for d in snapshot["required_decisions"]
                ),
            )
        )

    modifiers = modifier_labels(snapshot)
    facts.append(("Modifiers", ", ".join(modifiers) if modifiers else "None"))

    identity = snapshot.get("identity") or {}
    model = identity.get("model")
    provider = identity.get("provider")
    if model:
        facts.append(("Model", f"{model} ({provider})" if provider else str(model)))
    if snapshot.get("captured_at"):
        facts.append(("Generated", _format_timestamp(snapshot["captured_at"])))

    return tuple(facts)


def summary_line(snapshot: dict | None) -> str:
    """A single mobile-friendly line describing a snapshot's key choices."""
    facts = describe_inputs(snapshot)
    skip = {"Modifiers", "Model", "Generated", "Agent capabilities", "STRIDE threats"}
    parts = [value for label, value in facts if label not in skip]
    modifiers = modifier_labels(snapshot or {})
    if modifiers:
        parts.append(" + ".join(modifiers))
    return " · ".join(parts)


# Fields that identify the *user's choices*. Anything a page re-derives on each
# rerun (the per-phase technique sample, the resolved kill-chain string) is left
# out, or every rerun would look like an edit.
_COMPARABLE_FIELDS = (
    "scenario_type",
    "matrix",
    "organisation",
    "selected_entity",
    "selected_techniques",
    "selected_categories",
    "selected_stride",
    "selected_capabilities",
    "scenario_seed",
    "required_decisions",
    "template_info",
    "modifiers",
)


def comparable_inputs(snapshot: dict | None) -> dict:
    """The user-chosen subset of a snapshot, for equality comparisons."""
    if not snapshot:
        return {}
    return {
        field_name: snapshot[field_name]
        for field_name in _COMPARABLE_FIELDS
        if field_name in snapshot
    }


def inputs_changed(persisted: dict | None, current: dict | None) -> bool:
    """Has the form moved on from the inputs that produced the shown result?"""
    if not persisted or not current:
        return False
    return comparable_inputs(persisted) != comparable_inputs(current)
