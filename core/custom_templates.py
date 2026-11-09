"""Incident-response templates for the Custom Scenarios page.

Each template is a list of ``"Name (ID)"`` technique labels per matrix. The
labels must match an active technique in the shipped bundle, which
``tests/test_custom_templates.py`` checks, so a data refresh that renames or
revokes one fails CI instead of shrinking a template on the page.
"""

from __future__ import annotations

from core import attack_data

INCIDENT_RESPONSE_TEMPLATES = {
    "Enterprise": {
        "Phishing Attack": ["Spearphishing Attachment (T1566.001)", "User Execution (T1204)", "Software Extensions (T1176)", "Credentials from Password Stores (T1555)", "Input Capture (T1056)", "Exfiltration Over C2 Channel (T1041)"],
        "Ransomware Attack": ["Exploit Public-Facing Application (T1190)", "Windows Management Instrumentation (T1047)", "Create Account (T1136)", "Process Injection (T1055)", "Data Encrypted for Impact (T1486)"],
        "Malware Infection": ["Supply Chain Compromise (T1195)", "Command and Scripting Interpreter (T1059)", "Registry Run Keys / Startup Folder (T1547.001)", "Obfuscated Files or Information (T1027)", "Remote Services (T1021)", "Data Destruction (T1485)"],
        "Insider Threat": ["Valid Accounts (T1078)", "Account Manipulation (T1098)", "Exploitation for Privilege Escalation (T1068)", "Data Staged (T1074)", "Scheduled Transfer (T1029)", "Account Access Removal (T1531)"],
        "Cross-Site Scripting (XSS) Attack": ["Exploit Public-Facing Application (T1190)", "User Execution (T1204)", "Input Capture (T1056)", "Exfiltration Over Web Service (T1567)"],
        "SQL Injection Attack": ["Exploit Public-Facing Application (T1190)", "Exploitation for Credential Access (T1212)", "Exfiltration Over Web Service (T1567)"],
        "API Compromise": ["Active Scanning (T1595)", "Exploit Public-Facing Application (T1190)", "Exploitation for Client Execution (T1203)", "Valid Accounts (T1078)", "Application Layer Protocol (T1071)", "Data Manipulation (T1565)"],
    },
    "ICS": {
        "Remote Access Exploitation": ["External Remote Services (T0822)", "Exploit Public-Facing Application (T0819)", "Remote Services (T0886)", "Wireless Compromise (T0860)"],
        "ICS Data Manipulation": ["Modify Controller Tasking (T0821)", "Manipulate I/O Image (T0835)", "Modify Alarm Settings (T0838)", "Modify Parameter (T0836)"],
        "Denial of Service": ["Denial of Service (T0814)", "Activate Firmware Update Mode (T0800)", "Brute Force I/O (T0806)", "Command Message (T1691.001)"],
        "ICS Reconnaissance": ["Network Sniffing (T0842)", "Remote System Discovery (T0846)", "Network Connection Enumeration (T0840)", "Program Upload (T0845)"],
    },
    "ATLAS": {
        "Model Evasion Attack": ["Evade AI Model (AML.T0015)", "Craft Adversarial Data (AML.T0043)", "Black-Box Optimization (AML.T0043.001)", "Verify Attack (AML.T0042)"],
        "Data Poisoning Attack": ["Poison Training Data (AML.T0020)", "Publish Poisoned Datasets (AML.T0019)", "Poison AI Model (AML.T0018.000)", "AI Supply Chain Compromise (AML.T0010)"],
        "Model Extraction Attack": ["Discover AI Artifacts (AML.T0007)", "AI Model Inference API Access (AML.T0040)", "Exfiltration via AI Inference API (AML.T0024)", "Create Proxy AI Model (AML.T0005)"],
        "Prompt Injection Attack": ["Acquire Public AI Artifacts (AML.T0002)", "LLM Prompt Injection (AML.T0051)", "AI Agent Tool Invocation (AML.T0053)", "Exfiltration via AI Inference API (AML.T0024)"],
        "LLM Jailbreak": ["LLM Jailbreak (AML.T0054)", "LLM Prompt Injection (AML.T0051)", "LLM Prompt Crafting (AML.T0065)", "Spearphishing via Social Engineering LLM (AML.T0052.000)"],
        "AI Supply Chain Attack": ["AI Supply Chain Compromise (AML.T0010)", "Publish Poisoned Datasets (AML.T0019)", "Manipulate AI Model (AML.T0018)", "Publish Poisoned Models (AML.T0058)"],
    },
}


def resolve_incident_template(matrix: str, template: str) -> tuple[list[str], list[str]]:
    """Split a template's labels into ``(kept, dropped)``, both in template order.

    ``kept`` are labels that name an active technique in the matrix's bundle.
    ``dropped`` are the rest, so the page can say which ones it left out.
    Raises ``ValueError`` for an unknown matrix or template.
    """
    templates = INCIDENT_RESPONSE_TEMPLATES.get(matrix, {})
    if template not in templates:
        raise ValueError(f"Template '{template}' not found in the {matrix} matrix.")
    active = attack_data.active_technique_labels(matrix)
    kept = [label for label in templates[template] if label in active]
    dropped = [label for label in templates[template] if label not in active]
    return kept, dropped
