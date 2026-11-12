"""The MITRE matrices AttackGen supports, and the per-matrix facts callers read.

The sidebar offers ``MATRIX_NAMES`` in order, ``core.state`` accepts only these
names from a URL, ``mcp_server`` builds its ``Matrix`` type from them, and the
pages take their labels from ``MATRICES`` instead of comparing the matrix to a
string. A matrix's ``name`` is the value stored in
session state and the URL and passed to ``core.attack_data``, the prompts, the
Navigator layer and the detection report.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixInfo:
    name: str
    framework: str
    """``"ATT&CK"`` or ``"ATLAS"``, as in "Select ATT&CK Techniques"."""
    label: str
    """The framework as a result's "Generated from" facts name it."""
    technique_example: str
    """A technique ID and name the Custom page's search hint quotes."""
    case_studies: bool
    """Scenarios come from ATLAS case studies (the full documented procedure)
    rather than ATT&CK threat groups (one sampled technique per phase)."""

    @property
    def entity(self) -> str:
        """What page 1 asks the user to select."""
        return "case study" if self.case_studies else "threat actor group"


MATRICES: dict[str, MatrixInfo] = {
    matrix.name: matrix
    for matrix in (
        MatrixInfo(
            name="Enterprise",
            framework="ATT&CK",
            label="Enterprise ATT&CK",
            technique_example="`T1556` or `Phishing`",
            case_studies=False,
        ),
        MatrixInfo(
            name="ICS",
            framework="ATT&CK",
            label="ICS ATT&CK",
            technique_example="`T0814` or `Denial of Service`",
            case_studies=False,
        ),
        MatrixInfo(
            name="ATLAS",
            framework="ATLAS",
            label="MITRE ATLAS",
            technique_example="`AML.T0051` or `Prompt Injection`",
            case_studies=True,
        ),
    )
}

MATRIX_NAMES: tuple[str, ...] = tuple(MATRICES)
DEFAULT_MATRIX = MATRIX_NAMES[0]
