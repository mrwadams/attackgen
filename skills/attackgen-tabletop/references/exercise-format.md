# Tabletop / MSEL output format

Render the exercise in the following structure and house style. This is the
presentation contract for the `attackgen-tabletop` skill. All attack/detection
facts must come from the AttackGen MCP tool outputs (see `SKILL.md`).

## House style

- **British English** throughout (organisation, prioritise, defence, analyse).
- Realistic for the stated **industry** and **company size**. Give the target a
  plausible name and estate (identity model, key SaaS, notable third parties).
- Match the **actor's tradecraft**: espionage actors are patient and low-and-slow
  and lean on identity/cloud abuse; smash-and-grab actors move fast and loud.
- Every inject and discussion question references a specific technique — by name
  and ID — drawn from the resolved kill chain.
- Format in Markdown: headings, a kill-chain table, bold for key artefacts (e.g.
  `Add-MailboxPermission`), and inline event IDs where the detection data gives
  them (e.g. Event 4769).

## Section structure

Use these numbered sections, in order:

**Header** — a title line plus a short metadata block: Framework, Threat
actor (with aliases), Target profile (industry + size + estate), Exercise type
(purple-team tabletop / functional IR test).

**1. Scenario Narrative** — 2–3 paragraphs: who the target is, how the actor
works its way through the kill chain at a high level, and what capability the
exercise is testing. Save the step-by-step for the timeline (section 3).

**2. Kill Chain Walk-Through** — a Markdown table with columns
`# | ATT&CK Phase | Technique (ID) | Scenario action`, one row per technique
from `get_kill_chain`, in kill-chain phase order. The "Scenario action"
translates each technique into what the actor concretely does in this estate.

**3. Timeline of Events (Inject Schedule)** — the MSEL core. A sequence of
timestamped injects (`Day N — HH:MM — …`), each tied to a technique ID, moving
from earliest/low-signal (recon) to the exfiltration/impact that triggers
response. Mark **one inject as the exercise start point** ("← IR team starts
here") — usually the first inject a defender would realistically notice.

**4. Discussion Questions (by IR phase — NIST SP 800-61)** — group questions
under **Detection & Analysis**, **Containment**, **Eradication & Recovery**, and
**Post-Incident**. Make them specific to this kill chain and estate (e.g. "do you
have near-real-time visibility of `Add-MailboxPermission` (T1098.002)?"). Include
regulatory/notification questions where the industry implies them (e.g.
ICO/GDPR 72-hour, FCA, sector regulators).

**5. Detection & Response Reference** — build this from the
`get_detection_report` markdown. Lead with a consolidated **"Log sources to
enable"** line, then per technique give the detection strategy (DET id), its key
analytics + log sources, and the associated mitigations (M-codes). Frame it as a
scoring aid: "use this to score detection coverage during the exercise". Keep
recon/resource-development techniques as pre-compromise context.

**6. Success Criteria** — a short checklist of what a passing IR team
demonstrates: correlating the final signal back to the entry vector, enumerating
and revoking every attacker-added persistence artefact, triggering the correct
regulatory clocks, and producing a defensible containment plan.

## Optional add-ons (offer when relevant)

- An **ATT&CK Navigator layer** (from `get_navigator_layer`) to visualise the
  kill chain and score coverage.
- A **facilitator cut** vs. **participant cut** (hide the kill-chain table and
  detection reference from participants until the hotwash).
- An **AI-enhanced adversary** variant — the same kill chain reframed as
  AI-accelerated. If the user wants a model-authored version end-to-end, point
  them at the AttackGen MCP `generate_threat_group_scenario` tool.
