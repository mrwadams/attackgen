"""The scenario request: what a scenario page asks the coordinator to generate.

Each of the three scenario pages builds one :class:`ScenarioRequest` from its
widgets. The fields every page shares — framework, organisation, the selected
entity, the techniques the scenario is built from and the generation modifiers
— sit on the request itself; everything else a page needs lives on a per-type
payload (:class:`ThreatGroupPayload`, :class:`CustomPayload`,
:class:`AIInsiderPayload`). The coordinator (:mod:`core.scenario_page`) stamps
the scenario type, capture time and identity on at Generate time, persists the
request with the result, and hands it to the page's prompt, layer and detection
callbacks. The summaries (:mod:`core.summary`) and the Assistant read it
through attributes.

Adding a fourth kind of scenario means adding a payload here, not teaching
every consumer a new set of keys.

The request is frozen so a captured run cannot be edited after Generate: a
base-phase retry replays exactly what was captured.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar


@dataclass(frozen=True)
class Organisation:
    """The organisation profile a scenario is written for."""

    industry: str | None = None
    company_size: str | None = None


@dataclass(frozen=True)
class SelectedEntity:
    """What the scenario is built around: a group, case study, template or archetype."""

    type: str
    name: str | None


@dataclass(frozen=True)
class Modifiers:
    """The generation modifiers that were on when the request was captured."""

    ai_uplift: bool = False
    purple_team_narrative: bool = False
    template: str | None = None


@dataclass(frozen=True)
class RequestIdentity:
    """Who ran the request, and under which trace and filename; stamped by the coordinator."""

    page_id: str
    trace_name: str
    trace_tags: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    download_name: str | None = None


@dataclass(frozen=True)
class ScenarioPayload:
    """A scenario type's own inputs, beyond the fields every request shares."""

    rederived: ClassVar[frozenset[str]] = frozenset()
    """Fields — shared or payload — the page re-derives on every rerun.

    They describe *this run*, not the user's choices, so they are left out of
    :meth:`ScenarioRequest.choices`; otherwise every rerun would read as an edit.
    """

    def choices(self) -> dict[str, Any]:
        """This payload's user-chosen fields, for equality comparisons."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self.rederived
        }

    def facts(self) -> tuple[tuple[str, str], ...]:
        """Human-readable ``(label, value)`` facts about this payload's inputs."""
        return ()


@dataclass(frozen=True)
class ThreatGroupPayload(ScenarioPayload):
    """Page 1: the kill chain resolved for a threat group or ATLAS case study.

    For ATT&CK matrices the page samples one technique per phase on every
    rerun, so the request's ``techniques``, the kill-chain string built from
    them and their IDs are all re-derived — the group (the ``selected_entity``)
    is the user's choice.
    """

    kill_chain_string: str
    technique_ids: tuple[str, ...] = ()

    rederived: ClassVar[frozenset[str]] = frozenset(
        {"techniques", "kill_chain_string", "technique_ids"}
    )


@dataclass(frozen=True)
class CustomPayload(ScenarioPayload):
    """Page 2: the framing line for an optional incident-response template."""

    template_info: str = ""


@dataclass(frozen=True)
class AIInsiderPayload(ScenarioPayload):
    """Page 3: the insider-threat selections that stand in for a MITRE matrix."""

    selected_categories: tuple[str, ...] = ()
    selected_stride: tuple[str, ...] = ()
    selected_capabilities: tuple[str, ...] = ()
    scenario_seed: str = ""
    required_decisions: tuple[str, ...] = ()

    def facts(self) -> tuple[tuple[str, str], ...]:
        facts: list[tuple[str, str]] = []
        for values, label in (
            (self.selected_categories, "Threat categories"),
            (self.selected_stride, "STRIDE threats"),
            (self.selected_capabilities, "Agent capabilities"),
        ):
            if values:
                facts.append((label, ", ".join(str(v) for v in values)))
        if self.scenario_seed:
            facts.append(("Scenario seed", "Provided"))
        if self.required_decisions:
            facts.append(
                (
                    "Required decisions",
                    "; ".join(
                        str(d).split(" — ", 1)[0] for d in self.required_decisions
                    ),
                )
            )
        return tuple(facts)


# Shared fields that identify the user's choices. Identity and capture time
# describe the run, not the form, so they are never compared.
_CHOICE_FIELDS = (
    "scenario_type",
    "matrix",
    "organisation",
    "selected_entity",
    "techniques",
    "modifiers",
)


@dataclass(frozen=True)
class ScenarioRequest:
    """One scenario page's Generate-time inputs.

    ``techniques`` holds the techniques the scenario is built from, in the
    page's own form (page 1's kill-chain records, page 2's display labels); its
    length is the technique count. ``modifiers`` is ``None`` when the page
    captures none, in which case the coordinator falls back to its own
    ``defense_narrative`` flag. ``scenario_type``, ``identity`` and
    ``captured_at`` are stamped by the coordinator when Generate is pressed.
    """

    organisation: Organisation | None = None
    scenario_type: str | None = None
    matrix: str | None = None
    selected_entity: SelectedEntity | None = None
    techniques: tuple[Any, ...] = ()
    modifiers: Modifiers | None = None
    payload: ScenarioPayload | None = None
    identity: RequestIdentity | None = None
    captured_at: str | None = None

    @property
    def technique_count(self) -> int:
        return len(self.techniques)

    def choices(self) -> dict[str, Any]:
        """The user-chosen subset of this request, for equality comparisons.

        Anything the payload marks as re-derived on each rerun is left out, so
        page 1's per-phase resample and re-resolved kill chain never read as an
        edit.
        """
        rederived = self.payload.rederived if self.payload else frozenset()
        chosen = {
            name: getattr(self, name)
            for name in _CHOICE_FIELDS
            if name not in rederived
        }
        chosen["payload"] = self.payload.choices() if self.payload else None
        return chosen

    def stamped(
        self, *, scenario_type: str, identity: RequestIdentity, captured_at: str
    ) -> ScenarioRequest:
        """This request with the coordinator's stamps filled in where missing."""
        return replace(
            self,
            scenario_type=self.scenario_type or scenario_type,
            identity=self.identity or identity,
            captured_at=self.captured_at or captured_at,
        )
