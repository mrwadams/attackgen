"""Readiness as data: every reason a page cannot generate yet, in one list.

A scenario page has two kinds of requirement — the shared Setup values
(provider, credential, model, matrix, industry, size) that
:mod:`core.sidebar` resolves, and its own selection (a threat group, a
technique, a threat category). Both used to be checked *after* the Generate
click, as side effects that printed a message and returned a bool, so the user
paid a click to discover them and the page-specific one could be reported last.

:func:`resolve_readiness` combines the two into one immutable value. The page
supplies its own requirements, the coordinator derives the button's disabled
state and the visible list of blockers from the result, and no model call can
be reached while any blocker stands.

Page requirements come first: the shared Setup fields are the same on every
page and stated in the sidebar, whereas the missing selection is the thing the
user is looking at.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Readiness:
    """Whether generation can start, and every reason it can't."""

    page_blockers: tuple[str, ...] = ()
    setup_blockers: tuple[str, ...] = ()

    @property
    def blockers(self) -> tuple[str, ...]:
        """All blockers, page-specific first."""
        return self.page_blockers + self.setup_blockers

    @property
    def ready(self) -> bool:
        return not self.blockers


Requirements = Sequence[str] | Callable[[], Iterable[str]] | None
"""A page's own blockers: a sequence, or a callable returning one."""


def _as_tuple(requirements: Requirements) -> tuple[str, ...]:
    if requirements is None:
        return ()
    if callable(requirements):
        requirements = requirements()
    return tuple(str(item) for item in requirements if item)


def resolve_readiness(
    *,
    requirements: Requirements = None,
    setup=None,
) -> Readiness:
    """Combine a page's own requirements with the shared Setup blockers.

    ``setup`` is a :class:`core.sidebar.SetupState` (or anything exposing
    ``blockers``); it is optional so callers that only have page-specific
    requirements — and the coordinator's own default — behave sensibly.
    """
    return Readiness(
        page_blockers=_as_tuple(requirements),
        setup_blockers=tuple(getattr(setup, "blockers", ()) or ()),
    )
