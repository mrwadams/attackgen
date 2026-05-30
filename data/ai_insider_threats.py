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

AI_INSIDER_TEMPLATES = {
    "Rogue Coding Agent — Supply Chain Backdoor": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Supply Chain Sabotage"],
        "stride": ["T1", "T2", "S2", "E3"],
    },
    "Autonomous Agent Data Heist": {
        "archetype": "Human-as-Auditor (L4 — Critical Threat)",
        "categories": ["Data Exfiltration"],
        "stride": ["I1", "I2", "I3", "I4"],
    },
    "Credential Harvesting & Lateral Movement": {
        "archetype": "Human-as-Collaborator (L2 — Medium Threat)",
        "categories": ["Credential Compromise"],
        "stride": ["S1", "E2", "E4"],
    },
    "Infrastructure Sabotage & Denial of Service": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Infrastructure Sabotage"],
        "stride": ["D2", "D4", "T4"],
    },
    "Covert Agent — Evading Oversight": {
        "archetype": "Human-as-Auditor (L4 — Critical Threat)",
        "categories": ["Deception & Evasion"],
        "stride": ["S4", "R1", "R2", "E1"],
    },
    "Container Escape & Privilege Escalation": {
        "archetype": "Human-as-Supervisor (L3 — High Threat)",
        "categories": ["Credential Compromise", "Infrastructure Sabotage"],
        "stride": ["T4", "E1", "E2", "E3"],
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
