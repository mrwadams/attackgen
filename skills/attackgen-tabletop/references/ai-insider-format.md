# AI insider-threat tabletop format (Path B)

Render a Path B exercise in the structure below. This is the presentation
contract for the AI insider-threat path of the `attackgen-tabletop` skill.

`references/exercise-format.md` still governs the **house style** (British
English, realistic for the industry and size, every inject tied to a specific
identifier, Markdown with headings and tables). This file carries only what
differs, because the adversary is the organisation's own agent rather than an
external actor:

- There is **no kill chain and no ATT&CK ID**. The spine is the **STRIDE threat
  scope** (`S1`, `E5`, …) from the AI insider-threat model, plus the deployment
  archetype that governs what the agent could reach and what would be seen.
- The exercise tests **agent governance** as much as intrusion response: who can
  halt an agent fleet, whose identity the agent acts under, what survives when
  agent-side telemetry doesn't.
- The adversary has **no motive**. Behaviour follows from configuration, access,
  capability and objective — see the *no human motive* rule in `SKILL.md`.

All facts come from the `get_ai_insider_prompt` payload (and
`list_ai_insider_options`). The prompt's own seven sections are a *scenario*;
the sections below are the *exercise*. Map them across rather than emitting the
prompt's structure verbatim.

## Section structure

Use these numbered sections, in order.

**Header** — a title line plus a metadata block: Framework (AI Insider Threat
Model — *Actions Speak Louder Than Tokens*), Deployment archetype (with its
autonomy level, e.g. *Human-as-Auditor (L4 — Critical Threat)*), Threat scope
(the threat categories plus the STRIDE IDs in play), Target profile (industry +
size + estate, including the agent's tooling and identity model), Exercise type
(purple-team tabletop / functional IR test).

**1. Scenario Narrative** — 2–3 paragraphs from the prompt's *Scenario Title &
Overview* and *Deployment Context*: who the organisation is, what the agent was
deployed to do, how its work turns into an incident, and what capability the
exercise is testing. Name the agent and its platform (fictional but plausible).
Save the step-by-step for the inject schedule.

**2. Agent Profile & Threat Scope** *(facilitator)* — replaces Path A's
kill-chain table. Two parts:

- An **archetype block** quoting the four fields the tool supplies for the
  selected archetype: access, detection posture, primary threats at this level,
  and the critical control at this level. This is what makes the scenario
  plausible — an L1 agent cannot do what an L4 agent can.
- A **threat scope table**, one row per STRIDE threat in scope:
  `# | STRIDE (ID) | Threat | Scenario action` — the last column translating the
  threat into what this agent concretely does in this estate. Keep the model's
  ID and name verbatim; write only the scenario action yourself.

**3. Inject Schedule** — the MSEL core: timestamped injects
(`Day N — HH:MM — …`), each tagged with the STRIDE ID it exercises, running from
the agent's earliest legitimate-looking activity to the signal that triggers
response. Two things to get right on this path:

- **Blend malicious and legitimate work.** Most injects should be indistinguishable
  from the agent doing its job — that is the whole difficulty of agent insider
  threat. Reserve the obviously wrong actions for late in the timeline.
- **Start the IR team from the realistic signal.** Mark one inject
  "← IR team starts here". Where the scope includes telemetry suppression or an
  assumed containment boundary, the first defender-visible signal is usually
  **external** — a partner's report, a peer-side log, an outbound-traffic
  anomaly, a supplier's security team — not an agent-side alert. Set the start
  point accordingly, and leave the earlier injects as facilitator-only
  ground truth for the hotwash.

**4. Discussion Questions (by IR phase — NIST SP 800-61)** — group under
**Detection & Analysis**, **Containment**, **Eradication & Recovery**, and
**Post-Incident**.

- If the selections came from a template, its **`required_decisions` are
  mandatory**: every one must be forced by at least one question, phrased for
  this estate. They arrive as `Lead — detail`; the lead names the decision
  (e.g. *Containment*, *Identity rotation*, *Third-party notification*,
  *Evidence preservation*) and the detail says what must be decided. Distribute
  them to the phase each belongs to rather than listing them together, and don't
  drop the others — 5–8 questions total.
- Add the questions this path exists to force: who has the authority and the
  mechanism to halt an agent (or the whole fleet), and what breaks if they do;
  which identity the agent acted under and how a workload credential is rotated
  while dependent jobs are running; what the organisation owes an affected
  outside party and on what clock; what evidence survives if the agent's own
  logs are absent or untrustworthy; how you establish the blast radius when the
  agent's actions are spread across thousands of legitimate ones.
- Include regulatory/notification questions where the industry implies them
  (ICO/GDPR 72-hour, FCA, sector regulators), and contractual notification where
  a third party is affected.

**5. Detection & Response Reference** *(facilitator)* — build this from the
detection strategies and NIST CSF controls in the prompt payload, and score it
against the archetype's detection posture. Give:

- A **coverage table**: `STRIDE (ID) | Detection strategy | Would it fire here?`
  with a one-line justification per row. Be honest about the failures — the
  archetype's detection posture usually means several would not fire, and that
  is the finding the exercise is meant to surface.
- A **"telemetry to enable"** line consolidating what the organisation would
  need in place for the misses to become hits.
- **Recommended controls** mapped to the five NIST CSF functions (Identify,
  Protect, Detect, Respond, Recover), prioritised, drawn from the control set in
  the payload and led by the archetype's critical control.

Frame it as a scoring aid: use it to score detection coverage during the
exercise.

**6. Success Criteria** — a short checklist of what a passing team demonstrates:
containing the agent without destroying the evidence, distinguishing the agent's
sanctioned work from the incident, rotating the right identity in the right
order, establishing blast radius across systems the agent touched legitimately,
notifying the correct outside parties on the correct clock, and producing a
defensible account of what happened despite the telemetry gaps.

## Optional add-ons (offer when relevant)

- A **facilitator cut** vs. **participant cut** — hide sections 2 and 5 until
  the hotwash, as on Path A.
- A **re-run at a different archetype.** Rebuilding the same threat scope at
  L2 instead of L4 (or the reverse) makes the autonomy level the variable and
  shows the team how much of their exposure is a deployment decision.
- An **archetype-template re-run.** Templates without a `brief` invent a fresh
  premise on each generation, so the same threat profile yields a new exercise —
  useful for running the same team twice without them recognising it.
- If the user wants AttackGen's own model to author the scenario end-to-end
  rather than yours, point them at the MCP `generate_ai_insider_scenario` tool
  or page 3 of the Streamlit app.
