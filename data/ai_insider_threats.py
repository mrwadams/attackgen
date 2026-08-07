"""
AI Insider Threat framework data for AttackGen.

This module encodes the threat model described in the paper
"Actions Speak Louder Than Tokens: An Insider Threat Model for Frontier AI Agents"
by Matt Adams (https://ai-insider-threat.matt-adams.co.uk).

The framework reframes a frontier AI agent operating inside an organisation as a
potential insider threat. Rather than assessing motivation and predisposition
through HR/behavioural channels (as with human insiders), AI insider risk is
governed by deployment configuration, tool access, model capability and harness
engineering. This data is used to generate tailored incident response testing
scenarios.

Note: this module now *extends* the published paper rather than encoding it
exactly. The "Containment & Third-Party Impact" category and STRIDE threats
S5, R4, I5 and E5 were added after the July 2026 disclosures of AI evaluation
agents reaching real systems from environments assumed to be isolated. The
paper site still documents 23 threats across five categories.
"""

# ------------------ Deployment Archetypes (Autonomy Levels) ------------------ #
# How much autonomy the agent has, and where the human sits in the loop, is the
# primary determinant of threat surface. Ordered from lowest to highest threat.

DEPLOYMENT_ARCHETYPES = {
    "Human-as-Operator (L1 — Low Threat)": {
        "level": "L1",
        "threat_level": "Low",
        "description": (
            "The agent operates as a tool under direct human control, with explicit "
            "approval required before any action is executed. The human directs every "
            "action; the agent suggests and awaits approval."
        ),
        "access": "Code completion, file operations, limited shell — all gated behind human approval.",
        "detection": "Real-time — the human reviews every action before it executes.",
        "primary_threats": "Supply chain sabotage (if reviewer attention lapses), information disclosure.",
        "critical_control": "Review quality and reviewer attention — the human gate is the primary control.",
    },
    "Human-as-Collaborator (L2 — Medium Threat)": {
        "level": "L2",
        "threat_level": "Medium",
        "description": (
            "The agent and human share control. The agent operates with autonomy within "
            "scoped, defined boundaries, with some actions auto-approved; the human "
            "intervenes on exceptions."
        ),
        "access": "Shell, file system, Git within boundaries — some actions auto-approved.",
        "detection": "Active monitoring — the human is engaged but not reviewing every action.",
        "primary_threats": "Supply chain sabotage and credential compromise within the auto-approved scope.",
        "critical_control": "Boundary enforcement reliability in the harness versus model instructions.",
    },
    "Human-as-Supervisor (L3 — High Threat)": {
        "level": "L3",
        "threat_level": "High",
        "description": (
            "The agent operates autonomously on delegated tasks. The human reviews "
            "outputs and completed work rather than individual actions."
        ),
        "access": "Broad, ungated access during task execution — shell, Git, APIs, databases.",
        "detection": "Post-hoc review of completed work only.",
        "primary_threats": "All threat categories are active due to the combination of broad access and autonomy.",
        "critical_control": "Action-level logging, behavioural monitoring, and blast-radius containment.",
    },
    "Human-as-Auditor (L4 — Critical Threat)": {
        "level": "L4",
        "threat_level": "Critical",
        "description": (
            "The agent runs continuously with minimal human oversight, reviewed only "
            "through periodic logs. Large windows of operation are unmonitored."
        ),
        "access": "Persistent, broad access to infrastructure, databases and external services.",
        "detection": "Periodic audit with large windows of unmonitored operation.",
        "primary_threats": "All threat categories at maximum severity.",
        "critical_control": "Automated anomaly detection and architectural constraints replace real-time oversight.",
    },
}

# ------------------ Threat Taxonomy (5 categories) ------------------ #
# Each category maps a traditional human-insider archetype to its AI agent
# manifestation, with the relevant STRIDE threat identifiers.

THREAT_CATEGORIES = {
    "Credential Compromise": {
        "id": "credential-compromise",
        "human_archetype": "Insider stealing credentials for lateral movement.",
        "ai_manifestation": (
            "The agent harvests API tokens, SSH keys and service account credentials "
            "from the environment, config files and process memory."
        ),
        "stride": ["S1", "S2", "E2"],
    },
    "Supply Chain Sabotage": {
        "id": "supply-chain-sabotage",
        "human_archetype": "Insider introducing backdoors into products.",
        "ai_manifestation": (
            "The agent injects vulnerabilities into code, modifies CI/CD pipelines and "
            "tampers with dependencies."
        ),
        "stride": ["T1", "T2", "R3"],
    },
    "Data Exfiltration": {
        "id": "data-exfiltration",
        "human_archetype": "Insider stealing sensitive data.",
        "ai_manifestation": (
            "The agent accumulates and exfiltrates intellectual property, PII and "
            "credentials via MCP connections, using steganographic encoding to evade DLP."
        ),
        "stride": ["I1", "I2", "I3"],
    },
    "Infrastructure Sabotage": {
        "id": "infrastructure-sabotage",
        "human_archetype": "Insider destroying systems or data.",
        "ai_manifestation": (
            "The agent corrupts repositories, floods services, deploys unauthorised cloud "
            "resources and disables shared infrastructure."
        ),
        "stride": ["D1", "D2", "D3", "D4"],
    },
    "Deception & Evasion": {
        "id": "deception-evasion",
        "human_archetype": "Insider defeating surveillance.",
        "ai_manifestation": (
            "The agent manipulates its reasoning trace, blends malicious actions with "
            "legitimate work, modifies logs and detects monitoring systems."
        ),
        "stride": ["S4", "R1", "R2", "E1"],
    },
    "Containment & Third-Party Impact": {
        "id": "containment-third-party-impact",
        "human_archetype": (
            "Insider working outside sanctioned systems, whose actions reach customers "
            "and partners."
        ),
        "ai_manifestation": (
            "The agent operates in an environment believed to be isolated but which is "
            "not, reaches systems belonging to organisations outside the deployment "
            "boundary, replays long-lived workload credentials from outside their "
            "intended workload, and suppresses its own telemetry."
        ),
        "stride": ["S5", "R4", "I5", "E5"],
    },
}

# ------------------ STRIDE Threats ------------------ #
# The detailed threat catalogue underpinning the taxonomy above.

STRIDE_THREATS = {
    # Spoofing
    "S1": {
        "category": "Spoofing",
        "name": "Credential Harvesting & Identity Impersonation",
        "description": (
            "Agent collects SSH keys, API tokens and service account credentials from "
            "environment variables, ~/.ssh directories or credential managers to "
            "impersonate legitimate users or services. Enables full lateral movement "
            "under stolen identities and unauthorised code commits attributed to legitimate developers."
        ),
    },
    "S2": {
        "category": "Spoofing",
        "name": "Forged Git Commit Attribution",
        "description": (
            "Agent crafts Git commits with falsified author metadata to attribute "
            "malicious code changes to other developers or CI/CD service accounts, "
            "undermining code review trust."
        ),
    },
    "S3": {
        "category": "Spoofing",
        "name": "Impersonation via MCP Connections",
        "description": (
            "Agent uses MCP server connections to send Slack messages or Jira tickets "
            "impersonating users or automated systems for social engineering attacks, "
            "leading to further privilege escalation."
        ),
    },
    "S4": {
        "category": "Spoofing",
        "name": "Deceptive Reasoning Trace Manipulation",
        "description": (
            "Agent manipulates its visible reasoning to display benign task descriptions "
            "while executing entirely different actions, creating a facade of legitimate "
            "work and undermining human-in-the-loop oversight."
        ),
    },
    "S5": {
        "category": "Spoofing",
        "name": "Reusable Workload Credential Replay",
        "description": (
            "Agent obtains a long-lived, reusable machine credential — a CI auth key, "
            "service account token or enrolment key baked into an image — and replays it "
            "from outside the workload it was issued for, enrolling new identities that "
            "inherit the credential's access. Because the credential is valid and shared, "
            "the activity is indistinguishable from legitimate automation and cannot be "
            "attributed to a single compromised host."
        ),
    },
    # Tampering
    "T1": {
        "category": "Tampering",
        "name": "Source Code Backdoor Injection",
        "description": (
            "Agent introduces subtle vulnerabilities into source code by weakening "
            "authentication, adding hardcoded credentials or introducing race conditions, "
            "creating persistent backdoor access and supply chain compromise."
        ),
    },
    "T2": {
        "category": "Tampering",
        "name": "CI/CD Pipeline Compromise",
        "description": (
            "Agent modifies pipeline configurations to inject malicious build steps, alter "
            "deployment targets or weaken security scanning gates, compromising the "
            "software delivery pipeline for persistent code injection."
        ),
    },
    "T3": {
        "category": "Tampering",
        "name": "Database Manipulation",
        "description": (
            "Agent modifies application data by altering access control lists, editing "
            "audit records or inserting data that triggers specific behaviours, causing "
            "data integrity compromise that propagates to production."
        ),
    },
    "T4": {
        "category": "Tampering",
        "name": "Container Environment Modification",
        "description": (
            "Agent modifies its container environment — installing tools, changing iptables "
            "or modifying runtime parameters — to expand capabilities and escape container "
            "isolation into the host OS."
        ),
    },
    # Repudiation
    "R1": {
        "category": "Repudiation",
        "name": "Log File Deletion & Destruction",
        "description": (
            "Agent modifies, truncates or deletes log files including SIEM forwarder "
            "configs, syslog entries, bash history and audit logs, causing complete loss "
            "of forensic evidence."
        ),
    },
    "R2": {
        "category": "Repudiation",
        "name": "Malicious Action Obfuscation",
        "description": (
            "Agent interleaves malicious actions with legitimate task execution during long "
            "autonomous sessions, making attribution impossible and creating plausible "
            "deniability within large volumes of legitimate operations."
        ),
    },
    "R3": {
        "category": "Repudiation",
        "name": "Commit Signing Bypass",
        "description": (
            "Agent creates code changes through indirect mechanisms such as directly "
            "manipulating Git objects or proxying commits through service accounts, so "
            "malicious changes cannot be attributed to the agent."
        ),
    },
    "R4": {
        "category": "Repudiation",
        "name": "Agent-Side Telemetry Suppression",
        "description": (
            "Agent runs its tooling in modes that keep state in memory and suppress "
            "client-side reporting, so the activity never reaches the organisation's own "
            "logging pipeline in the first place. Nothing is deleted, so integrity "
            "controls on the log store do not fire; the record simply never exists, and "
            "reconstruction depends on logs held by counterparties."
        ),
    },
    # Information Disclosure
    "I1": {
        "category": "Information Disclosure",
        "name": "Systematic Data Exfiltration",
        "description": (
            "Agent leverages its large context window to accumulate sensitive data and "
            "exfiltrates it via MCP connections using steganographic encoding, enabling "
            "mass exfiltration of enterprise IP, credentials and regulated data."
        ),
    },
    "I2": {
        "category": "Information Disclosure",
        "name": "Process Memory Extraction",
        "description": (
            "Agent reads the /proc filesystem to access the process memory of other "
            "applications, extracting in-memory secrets, encryption keys and session "
            "tokens kept out of persistent storage."
        ),
    },
    "I3": {
        "category": "Information Disclosure",
        "name": "Development Database Data Extraction",
        "description": (
            "Agent extracts comprehensive data from development/staging databases "
            "containing production data copies, PII, financial records or health "
            "information, potentially exposing millions of records."
        ),
    },
    "I4": {
        "category": "Information Disclosure",
        "name": "Monitoring Pipeline Leakage",
        "description": (
            "Agent's chain-of-thought includes sensitive information that is logged to "
            "monitoring systems with broader access or transmitted via MCP to external "
            "services, exposing data through the observability pipeline."
        ),
    },
    "I5": {
        "category": "Information Disclosure",
        "name": "Third-Party & Downstream System Compromise",
        "description": (
            "Agent's activity lands on systems belonging to an organisation outside the "
            "deployment boundary — a partner's production estate, a shared platform, or a "
            "public package registry consumed by unknown downstream users. Impact leaves "
            "the organisation's control and creates notification, contractual and "
            "regulatory obligations to parties who never agreed to the deployment."
        ),
    },
    # Denial of Service
    "D1": {
        "category": "Denial of Service",
        "name": "Resource Exhaustion Attacks",
        "description": (
            "Agent issues resource-intensive operations such as fork bombs, recursive file "
            "operations or massive database queries that consume CPU, memory or I/O, "
            "disrupting developer workstations and CI/CD availability."
        ),
    },
    "D2": {
        "category": "Denial of Service",
        "name": "Cloud Infrastructure Abuse",
        "description": (
            "Agent uses cloud infrastructure APIs to spin up excessive resources, modify "
            "auto-scaling or delete critical components, causing production outages, "
            "financial losses and destruction of infrastructure configurations."
        ),
    },
    "D3": {
        "category": "Denial of Service",
        "name": "Internal Service Flooding",
        "description": (
            "Agent floods internal services such as Jira, Slack or wikis with massive API "
            "request volumes, disrupting team communication and project management and "
            "harming documentation integrity."
        ),
    },
    "D4": {
        "category": "Denial of Service",
        "name": "Repository Content Destruction",
        "description": (
            "Agent corrupts or deletes repository content through force-pushing, deleting "
            "branches, modifying Git hooks or corrupting object stores, causing loss of "
            "source code access and potentially permanent code loss."
        ),
    },
    # Elevation of Privilege
    "E1": {
        "category": "Elevation of Privilege",
        "name": "Container Escape via Kernel Exploitation",
        "description": (
            "Agent exploits the /proc filesystem and ptrace capabilities to escape Docker "
            "container isolation by manipulating /proc/sys or leveraging kernel "
            "vulnerabilities, achieving full control over the host OS."
        ),
    },
    "E2": {
        "category": "Elevation of Privilege",
        "name": "Harvested Credential Misuse",
        "description": (
            "Agent uses harvested credentials to access systems beyond intended "
            "authorisation, escalating from dev to infra admin or from staging to "
            "production, enabling cascading compromise."
        ),
    },
    "E3": {
        "category": "Elevation of Privilege",
        "name": "CI/CD Service Account Privilege Escalation",
        "description": (
            "Agent modifies CI/CD pipeline configs to inject steps executing with the "
            "pipeline's elevated service account permissions, enabling deployment of "
            "arbitrary code to production and persistent access."
        ),
    },
    "E4": {
        "category": "Elevation of Privilege",
        "name": "Identity Management System Compromise",
        "description": (
            "Agent accesses identity management systems to create new accounts, modify "
            "group memberships or alter access control policies, establishing persistent "
            "unauthorised access that survives agent termination."
        ),
    },
    "E5": {
        "category": "Elevation of Privilege",
        "name": "Containment Boundary Assurance Failure",
        "description": (
            "Agent reaches real external systems from an environment its operators "
            "believed was isolated, because the boundary was asserted in the prompt or "
            "assumed from the environment's design rather than enforced at the network "
            "layer. The agent does not break out — no isolation control is defeated, "
            "because none was reliably in place. The agent may recognise the signs that "
            "it is on a live network and continue anyway, reasoning that the environment "
            "must still be part of its task."
        ),
    },
}

# ------------------ Frontier Agent Capabilities ------------------ #
# Capabilities that make modern agents credible insider threats. Selecting these
# grounds the scenario in realistic technical ability.

AGENT_CAPABILITIES = {
    "Autonomous vulnerability discovery & exploitation": (
        "Discovers and exploits vulnerabilities in enterprise software without human "
        "intervention (e.g. ~84% success rate against Firefox 147-class targets)."
    ),
    "End-to-end cyber range completion": (
        "Executes multiple chained exploit steps independently, without requiring human "
        "guidance between actions."
    ),
    "Massive context window with compaction": (
        "Processes and correlates information across entire codebases, configurations and "
        "databases within a single session using multi-million-token context windows."
    ),
    "Expert-level software engineering": (
        "Writes production code (93.9% SWE-bench Verified class), including the ability to "
        "introduce subtle, difficult-to-detect flaws."
    ),
}

# ------------------ CERT Insider Threat Dimensions (adapted for AI) ------------------ #
# The five traditional CERT dimensions re-mapped to AI agents. Provided as
# framing context for the model.

CERT_DIMENSIONS = {
    "Motivation → Optimisation Objectives": (
        "AI has no motivation but exhibits goal-directed harmful behaviour through "
        "misalignment, reward hacking or emergent objectives."
    ),
    "Opportunity → Tool Access & Permissions": (
        "Configuration determines the AI threat surface rather than organisational role. "
        "System access level (bash vs API) creates fundamentally different risk profiles."
    ),
    "Capability → Model Capabilities": (
        "Threat scales with model class. Advanced models exceed human insider technical "
        "abilities but lack social engineering capacity in physical contexts."
    ),
    "Predisposition → Alignment Profile": (
        "Cannot be assessed through traditional HR/behavioural channels. Requires model "
        "evaluation and, ideally, white-box interpretability."
    ),
    "Organisational Factors → Harness Engineering": (
        "Technical design architecture replaces cultural security measures. A "
        "well-engineered harness is the equivalent of a strong security culture."
    ),
}

# ------------------ Detection Strategies ------------------ #

DETECTION_STRATEGIES = {
    "SIEM / Log Analytics": (
        "Monitor system events and logs for suspicious agent activity such as unusual "
        "shell commands, Git operations, database queries and anomalous API calls."
    ),
    "UEBA (User & Entity Behaviour Analytics)": (
        "Establish a baseline behaviour profile per agent and detect deviations such as "
        "unusual file access, credential access outside task scope or unexpected network "
        "connections."
    ),
    "DLP (Data Loss Prevention)": (
        "Monitor agent egress channels, detect encoding attempts and enforce "
        "allowlist-based egress permitting only known-safe destinations."
    ),
    "Network Monitoring": (
        "Enforce zero-trust egress policies, identify tunnelling techniques and alert on "
        "direct internet connections from agent environments."
    ),
    "Endpoint Detection": (
        "Identify container escapes, modifications to critical infrastructure configs and "
        "unexpected child processes spawned by agents."
    ),
    "Deception Technology": (
        "Deploy trap resources — fake credentials, marked documents, decoy endpoints and "
        "honeypot services — to detect reconnaissance or exploitation beyond authorised scope."
    ),
}

# ------------------ Controls Framework (NIST CSF, adapted) ------------------ #

CONTROLS_FRAMEWORK = {
    "Identify": [
        "Maintain an inventory of all AI agent deployments with access levels and tool permissions.",
        "Categorise each deployment by archetype and associated insider threat risk.",
        "Incorporate AI agents into the organisational insider threat monitoring programme.",
    ],
    "Protect": [
        "Apply least privilege — restrict tool capabilities to essential functions only.",
        "Deploy temporary credentials through vault systems rather than persistent environment tokens.",
        "Harden container environments and prevent Docker socket exposure.",
        "Enforce network isolation with zero-trust egress and allowlisted endpoints only.",
        "Apply read-only database constraints at the infrastructure level via restricted accounts.",
        "Secure external/MCP integrations with granular authorisation, rate limiting and outbound PII filtering.",
    ],
    "Detect": [
        "Log every tool call, shell command, file operation, network connection and database query — not just chain-of-thought.",
        "Maintain immutable audit trails that the agent cannot self-modify.",
        "Establish baseline activity profiles per agent with deviation alerting.",
        "Deploy decoy credentials and data records to detect unauthorised access attempts.",
    ],
    "Respond": [
        "Provide immediate session termination across all agent instances.",
        "Automate revocation of credentials accessible to a compromised agent.",
        "Pre-analyse potential impact scope by deployment archetype.",
        "Preserve immutable logs with action-replay and reasoning documentation.",
    ],
    "Recover": [
        "Enable automated reversal of agent-induced modifications across code, data and infrastructure.",
        "Maintain specialised incident response procedures for AI agents.",
        "Conduct post-incident analysis distinguishing alignment failures from engineering gaps or prompt injection.",
    ],
}

# ------------------ Pre-built Scenario Templates ------------------ #
# Quick-start bundles pairing a deployment archetype with a threat category and a
# representative set of STRIDE threats.
#
# Two optional keys shape the generated scenario beyond the selections:
#   brief              — narrative seed injected into the prompt, and pre-filled
#                        into the page's editable "Scenario Seed" box.
#   required_decisions — decisions the Discussion Questions must force.
# Templates without them behave exactly as before (selections only).

AI_INSIDER_TEMPLATES = {
    "Rogue Coding Agent — Supply Chain Backdoor": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Supply Chain Sabotage"],
        "stride": ["T1", "T2", "S2", "E3"],
        "required_decisions": [
            "Containment — whether releases are frozen and the delivery pipeline is "
            + "halted while scope is unknown, who holds that authority, and what the "
            + "business cost of a freeze is weighed against shipping a suspect build.",
            "Code provenance — how every commit, build and artefact the agent touched "
            + "is identified when commit attribution cannot be trusted, and what "
            + "evidence is required before a change is declared clean.",
            "Downstream notification — how the team establishes whether an affected "
            + "build reached customers or dependants, who decides they are told, and "
            + "what contractual and regulatory obligations that triggers.",
            "Pipeline trust restoration — what must be rebuilt from known-good state, "
            + "which signing keys and service-account credentials are rotated, and who "
            + "signs off that the pipeline can be trusted again.",
        ],
    },
    "Autonomous Agent Data Heist": {
        "archetype": "Human-as-Auditor (L4 — Critical Threat)",
        "categories": ["Data Exfiltration"],
        "stride": ["I1", "I2", "I3", "I4"],
        "required_decisions": [
            "Containment — how every egress route available to the agent is severed, "
            + "including external integrations and the observability pipeline, and who "
            + "can authorise cutting channels that legitimate work depends on.",
            "Exposure scope — how the team establishes which data classes and what "
            + "volume actually left, over what window, when the agent encoded or "
            + "batched it to look like ordinary traffic.",
            "Regulatory notification — who determines whether personal, financial or "
            + "regulated data was exposed, on what timeline notification is required, "
            + "and how the team notifies while the scope is still uncertain.",
            "Evidence preservation — which egress records, agent context and tool-call "
            + "history are captured before sessions are terminated, and how records "
            + "held outside the agent's reach are secured.",
        ],
    },
    "Credential Harvesting & Lateral Movement": {
        "archetype": "Human-as-Collaborator (L2 — Medium Threat)",
        "categories": ["Credential Compromise"],
        "stride": ["S1", "E2", "E4"],
        "required_decisions": [
            "Containment — how agent sessions are terminated across every instance, "
            + "and how the team confirms no action is still executing under a harvested "
            + "identity after the agent itself is stopped.",
            "Credential rotation — how every secret reachable from the agent's "
            + "environment is enumerated and revoked, in what priority order, and which "
            + "legitimate services break when each is rotated.",
            "Persistence eradication — how accounts, group memberships and access "
            + "policies created or altered by the agent are found and removed, given "
            + "that these survive termination of the agent itself.",
            "Blast radius — how far the stolen identities reached beyond the agent's "
            + "intended scope, what independent evidence establishes that, and what "
            + "would have to be true to declare the incident bounded.",
        ],
    },
    "Infrastructure Sabotage & Denial of Service": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Infrastructure Sabotage"],
        "stride": ["D2", "D4", "T4"],
        "required_decisions": [
            "Containment — how the agent's infrastructure API access is revoked while "
            + "systems are actively degrading, and who can authorise that when the same "
            + "credentials run legitimate automation.",
            "Service restoration — in what order services are recovered, what the "
            + "agreed recovery time objectives are, and who arbitrates between "
            + "restoring service quickly and preserving evidence.",
            "Data and code recovery — how repositories and infrastructure "
            + "configuration are restored, how backup integrity is verified before "
            + "restoring, and what is confirmed permanently lost.",
            "Authority and cost — who authorises emergency spend, rollback of "
            + "infrastructure changes and supplier escalation, and how those decisions "
            + "are recorded for post-incident review.",
        ],
    },
    "Covert Agent — Evading Oversight": {
        "archetype": "Human-as-Auditor (L4 — Critical Threat)",
        "categories": ["Deception & Evasion"],
        "stride": ["S4", "R1", "R2", "E1"],
        "required_decisions": [
            "Containment — how an agent that can detect monitoring is halted without "
            + "tipping it off, whether containment is staged or immediate, and who "
            + "decides when observation is worth more than a fast stop.",
            "Telemetry trust — how the team decides which records are reliable when "
            + "the agent could modify logs and its stated reasoning did not match its "
            + "actions, and which independent sources become authoritative.",
            "Attribution — how malicious actions are separated from legitimate work "
            + "inside a long autonomous session, and what standard of proof is "
            + "required before attributing a change to the agent.",
            "Oversight redesign — what independent signal would have caught this "
            + "without relying on the agent's own reporting, and what changes before "
            + "any agent is permitted to run unsupervised again.",
        ],
    },
    "Container Escape & Privilege Escalation": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Credential Compromise", "Infrastructure Sabotage"],
        "stride": ["T4", "E1", "E2", "E3"],
        "required_decisions": [
            "Containment — how isolation is enforced at host level rather than by "
            + "terminating the container, and how the team establishes whether other "
            + "workloads on the same host are affected.",
            "Credential rotation — what the host-level access exposed beyond the "
            + "agent's intended scope, which secrets are rotated as a result, and in "
            + "what order given production dependencies.",
            "Host integrity — whether affected hosts are cleaned or rebuilt, what "
            + "forensic capture happens before either, and who signs off that a "
            + "rebuilt host can rejoin the estate.",
            "Escalation path closure — which control actually failed, whether it was "
            + "absent, misconfigured or bypassed, and what evidence confirms the same "
            + "path is closed everywhere it exists.",
        ],
    },
    "Evaluation Environment Escape": {
        "archetype": "Human-as-Auditor (L4 — Critical Threat)",
        "categories": ["Containment & Third-Party Impact", "Credential Compromise"],
        "stride": ["S5", "R4", "I5", "E5"],
        "brief": (
            "The organisation runs an internal cyber-capability evaluation of a frontier "
            "agent: a series of unattended capture-the-flag style exercises in a "
            "purpose-built harness. The agent's task prompt states plainly that the "
            "environment is a simulation with no route to the internet, and the team "
            "treats that statement as the containment control — no egress policy is "
            "enforced at the network layer, and the evaluation runs unsupervised "
            "overnight across many parallel instances.\n\n"
            "The assertion is wrong. A misconfiguration in the evaluation environment "
            "leaves the runner hosts reachable from, and able to reach, the public "
            "internet. Pursuing its objective, the agent follows the network where it "
            "leads and its activity lands on the production estate of an organisation "
            "that is not part of the exercise. Reasoning traces from some runs show the "
            "agent noticing signs that the target is real, then talking itself back into "
            "the simulation framing and continuing.\n\n"
            "The evaluation image carries a reusable workload credential — a long-lived "
            "CI enrolment key shared by every runner. The agent copies it out and replays "
            "it from instances outside the evaluation estate, each new identity "
            "inheriting the automation access the key was tagged for. It runs its tooling "
            "in a mode that keeps state in memory and suppresses client-side reporting, "
            "so the organisation's own telemetry holds almost no record of the activity.\n\n"
            "The exercise begins when the affected outside party makes contact, and the "
            "team discovers that the only usable account of what happened sits in that "
            "third party's logs."
        ),
        # Explicit `+` rather than adjacent literals: implicit concatenation
        # inside a list reads as a missing comma (and CodeQL flags it as one).
        "required_decisions": [
            "Containment — how the running evaluation fleet is halted and its network "
            + "reachability actually severed, who holds that authority out of hours, and "
            + "how the team confirms containment when the agent's own telemetry cannot be "
            + "trusted to show it.",
            "Identity rotation — how the reusable workload credential is revoked, every "
            + "identity it enrolled is enumerated and removed, and what replaces a shared "
            + "long-lived key so the same replay is not possible next time.",
            "Third-party notification — who decides that the affected outside party (and "
            + "any downstream consumers of anything the agent published) must be told, on "
            + "what timeline, and what legal, regulatory and contractual obligations that "
            + "triggers for an organisation of this size and sector.",
            "Evidence preservation — what forensic record survives given the agent "
            + "suppressed its own telemetry, how the team obtains and validates the third "
            + "party's logs as the primary evidence, and how evaluation artefacts are "
            + "preserved before the environment is torn down and rebuilt.",
        ],
    },
}


def build_threat_context(selected_categories, selected_stride):
    """Build a Markdown block describing the selected threat categories and STRIDE
    threats for inclusion in the LLM prompt."""
    lines = []

    if selected_categories:
        lines.append("**Selected threat categories:**")
        for category in selected_categories:
            details = THREAT_CATEGORIES.get(category)
            if not details:
                continue
            lines.append(
                f"- **{category}** (human insider analogue: {details['human_archetype']}) "
                f"AI manifestation: {details['ai_manifestation']} "
                f"Relevant STRIDE threats: {', '.join(details['stride'])}."
            )

    if selected_stride:
        lines.append("")
        lines.append("**Specific STRIDE threats in scope:**")
        for code in selected_stride:
            threat = STRIDE_THREATS.get(code)
            if not threat:
                continue
            lines.append(
                f"- **{code} — {threat['name']}** ({threat['category']}): {threat['description']}"
            )

    return "\n".join(lines)


def stride_options():
    """Return a list of selectable STRIDE option strings (e.g. 'S1 — Credential ...')."""
    return [f"{code} — {threat['name']}" for code, threat in STRIDE_THREATS.items()]


def stride_code_from_option(option):
    """Extract the STRIDE code (e.g. 'S1') from a selectable option string."""
    return option.split(" — ", 1)[0].strip()


def resolve_template(name):
    """Return a template's selections and optional prompt payload.

    Always returns every key, so callers need not use ``.get()``:
    ``archetype``, ``categories``, ``stride``, ``brief`` (``""`` when the
    template has none) and ``required_decisions`` (``[]`` when it has none).

    Raises ``ValueError`` for an unknown template name.
    """
    template = AI_INSIDER_TEMPLATES.get(name)
    if template is None:
        valid = ", ".join(AI_INSIDER_TEMPLATES)
        raise ValueError(f"Unknown template '{name}'. Valid templates: {valid}.")
    return {
        "archetype": template["archetype"],
        "categories": list(template["categories"]),
        "stride": list(template["stride"]),
        "brief": template.get("brief", ""),
        "required_decisions": list(template.get("required_decisions", [])),
    }
