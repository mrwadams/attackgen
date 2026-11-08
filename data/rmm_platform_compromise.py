"""Vendor-neutral RMM platform compromise tabletop preset.

The Custom Scenarios page normally needs only a name and a list of ATT&CK
techniques.  This preset also carries the incident premise and the decisions a
facilitator should force, because an RMM compromise is primarily a
blast-radius and trust-recovery problem rather than a single endpoint event.
"""

RMM_PLATFORM_COMPROMISE = "RMM Platform Compromise"

RMM_PLATFORM_COMPROMISE_TECHNIQUES = (
    "Valid Accounts (T1078)",
    "Remote Access Tools (T1219)",
    "Software Deployment Tools (T1072)",
    "PowerShell (T1059.001)",
    "Scheduled Task (T1053.005)",
    "Service Execution (T1569.002)",
    "Remote System Discovery (T1018)",
    "Service Stop (T1489)",
)

RMM_PLATFORM_COMPROMISE_BRIEF = (
    "An adversary has compromised the organisation's RMM platform and is "
    "abusing its privileged remote-management and software-deployment "
    "functions across managed endpoints. Treat the RMM control plane, its "
    "identities, automation, and affected endpoints as potentially untrusted."
)

RMM_PLATFORM_COMPROMISE_REQUIRED_DECISIONS = (
    "Privileged access — which RMM identities, API tokens, sessions, and "
    "integrations are revoked first, and who can authorise the action?",
    "Containment boundary — when is the RMM control plane disabled, and which "
    "endpoints or customer tenants are isolated independently?",
    "Scope — how are affected tenants, endpoints, commands, scripts, and "
    "deployment jobs identified without relying only on the compromised platform?",
    "Evidence — which control-plane audit records and endpoint artefacts are "
    "preserved before containment changes them?",
    "Notification — what evidence and severity thresholds trigger customer, "
    "insurer, regulator, or law-enforcement notification?",
    "Trusted recovery — how are clean administration paths, credentials, "
    "agents, policies, and automation restored in controlled stages?",
)


def rmm_platform_compromise_template_info() -> str:
    """Return the prompt context for the RMM compromise preset."""

    decisions = "\n".join(
        f"- {decision}" for decision in RMM_PLATFORM_COMPROMISE_REQUIRED_DECISIONS
    )
    return (
        f"This is an '{RMM_PLATFORM_COMPROMISE}' scenario.\n\n"
        f"Incident premise: {RMM_PLATFORM_COMPROMISE_BRIEF}\n\n"
        "The Discussion Questions must force decisions about:\n"
        f"{decisions}"
    )
