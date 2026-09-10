"""Tests for `core.readiness` — readiness expressed as data.

The contract is small and entirely observable: which blockers there are, in
which order, and whether generation may start at all.
"""

from __future__ import annotations

from core.readiness import Readiness, resolve_readiness
from core.sidebar import get_setup_state


def test_no_requirements_and_complete_setup_is_ready() -> None:
    setup = get_setup_state(
        {
            "chosen_model_provider": "OpenAI API",
            "llm_model_name": "gpt-5.5",
            "matrix": "Enterprise",
            "industry": "Finance / Banking",
            "company_size": "Medium (51-200 employees)",
        },
        {"OPENAI_API_KEY": "k"},
    )

    readiness = resolve_readiness(requirements=[], setup=setup)

    assert readiness.ready is True
    assert readiness.blockers == ()


def test_page_requirements_are_listed_before_setup_blockers() -> None:
    """The missing selection the user is looking at is not buried behind Setup."""
    setup = get_setup_state({"chosen_model_provider": "OpenAI API"}, {})

    readiness = resolve_readiness(
        requirements=lambda: ["Select a threat actor group for the scenario."],
        setup=setup,
    )

    assert readiness.ready is False
    assert readiness.blockers[0] == "Select a threat actor group for the scenario."
    assert readiness.blockers[1:] == setup.blockers
    assert readiness.page_blockers == ("Select a threat actor group for the scenario.",)


def test_requirements_accept_a_sequence_or_a_callable() -> None:
    from_sequence = resolve_readiness(requirements=["Pick something."])
    from_callable = resolve_readiness(requirements=lambda: ["Pick something."])

    assert from_sequence == from_callable == Readiness(page_blockers=("Pick something.",))


def test_empty_requirement_strings_are_ignored() -> None:
    readiness = resolve_readiness(requirements=["", None, "Real blocker."])

    assert readiness.blockers == ("Real blocker.",)


def test_missing_setup_means_only_the_page_requirements_apply() -> None:
    readiness = resolve_readiness(requirements=["Pick something."], setup=None)

    assert readiness.setup_blockers == ()
    assert readiness.ready is False
