---
name: attackgen-tabletop
description: >-
  Turn a MITRE ATT&CK/ATLAS threat group or case study — or a frontier AI agent
  deployed inside the organisation — into a full incident-response tabletop
  exercise (MSEL): scenario narrative, kill chain or agent threat scope,
  timestamped injects, discussion questions, and a detection-coverage scorecard,
  using the AttackGen MCP server. Use when the user wants a tabletop, MSEL, or
  purple-team IR exercise for a specific threat actor, ICS group, ATLAS case
  study, set of ATT&CK/ATLAS techniques, or an AI insider-threat / rogue-agent
  scenario (autonomous coding agent, agent data exfiltration, evaluation
  environment escape, agent supply-chain sabotage), or asks for one as a
  printable HTML report.
version: 1.1.0
license: GPL-3.0-or-later
---

# AttackGen Tabletop / MSEL

Turn AttackGen's data into a facilitator-ready tabletop exercise. This skill
orchestrates the AttackGen MCP tools and formats their output; the attack,
threat-model and detection data always comes live from those tools.

Two paths, sharing one output shape:

| Path | Adversary | Data source |
|------|-----------|-------------|
| **A — MITRE adversary** (default) | An external threat group or ATLAS case study | ATT&CK / ATLAS kill chain + detection join |
| **B — AI insider threat** | A frontier AI agent deployed *inside* the organisation | AI insider-threat model: deployment archetype, threat taxonomy, STRIDE threats |

## Prerequisites

The **AttackGen MCP server** must be connected. Tool names may be namespaced by
your client, e.g. `attackgen.get_kill_chain`.

Path A tools:

- `list_threat_groups(matrix)` / `list_case_studies()` — resolve a group or
  case-study name.
- `get_kill_chain(matrix, group, seed?, industry?, company_size?)` — the kill
  chain (one technique per phase for ATT&CK; full procedure for ATLAS).
- `get_detection_report(matrix, technique_ids)` — the Detection & Response join
  (ATT&CK v18 detection strategies + analytics + mitigations).
- `get_navigator_layer(matrix, technique_ids, name?, description?)` — optional
  Navigator layer JSON for a coverage visual.

Path B tools:

- `list_ai_insider_options()` — the option catalogue: `archetypes`,
  `threat_categories`, `stride`, `capabilities`, and quick-start `templates`
  (each with its `archetype`, `categories`, `stride`, `brief` and
  `required_decisions`).
- `get_ai_insider_prompt(template?, archetype?, categories?, stride?,
  capabilities?, industry, company_size, scenario_seed?)` — returns
  `{"messages": [system, user]}`, a composed prompt carrying the whole framework
  payload. Makes no LLM call and needs no key.

If these tools aren't connected, ask the user to add the server
(`claude mcp add attackgen -- python -m mcp_server`, or their client's
equivalent) and stop there.

## Choosing a path

Take **Path B** when the adversary is the organisation's *own* AI agent — an
autonomous coding or operations agent, an agent fleet, an evaluation harness, a
"rogue agent", or anything the user frames as AI insider risk. Take **Path A**
for a named external actor, an ICS group, an ATLAS case study, or a set of
ATT&CK/ATLAS techniques. If the user wants an external actor whose *tradecraft*
is AI-accelerated, that is still Path A — mention the AI-enhanced adversary
variant (`references/exercise-format.md`) rather than switching paths.

## Workflow — Path A (MITRE adversary)

1. **Gather inputs:** `matrix` (`Enterprise`/`ICS`/`ATLAS`), the `group` (threat
   group or ATLAS case-study name), `industry`, and `company_size`. Ask for
   whatever is missing. If unsure of a group name, call
   `list_threat_groups(matrix)` / `list_case_studies()` and pass the matched name
   verbatim.

2. **Resolve the kill chain:** call `get_kill_chain` with `matrix`, `group`,
   `industry`, `company_size`. Pass a `seed` (any integer) for a repeatable
   exercise; omit it for a fresh random draw. Read `techniques` (`Technique
   Name`, `ATT&CK ID`, `Phase Name`) and `kill_chain_string`.

3. **Pull detection data:** call `get_detection_report` with `matrix` and the
   `ATT&CK ID`s from step 2; use its `markdown` as the Detection & Response
   section. When it is `null` (ATLAS has mitigations only), say so and keep
   detection guidance light.

4. **(Optional) Coverage layer:** for a visual, call `get_navigator_layer` with
   the same IDs and offer the `layer_json` as a downloadable Navigator layer.

5. **Choose the format and author the exercise** (see *Output format* below),
   following `references/exercise-format.md` and tying every inject and
   discussion question to the techniques from step 2 and the detection data from
   step 3.

## Workflow — Path B (AI insider threat)

1. **Gather inputs:** `industry` and `company_size` are required. You also need
   *either* a `template` *or* an `archetype` — supplying neither is an error.

2. **Read the catalogue:** call `list_ai_insider_options()`. Offer the user the
   `templates` by name, summarising each from its `brief` (where it has one) and
   its `required_decisions`. Two kinds:
   - **Archetype templates** (no `brief`) — decisions only, so the model invents
     the premise; re-running one gives a fresh exercise.
   - **Incident templates** (with a `brief`) — pin a specific situation, so the
     narrative is stable across runs.

   With no template, take an `archetype` (autonomy level L1–L4) plus one or more
   `categories`, and optionally narrow to specific `stride` codes. Pass names
   verbatim from the catalogue.

3. **Build the prompt:** call `get_ai_insider_prompt` with the selections plus
   `industry` and `company_size`. A template fills gaps and anything you pass
   explicitly wins over it; `scenario_seed` (free text describing the situation
   to rehearse) replaces a template's `brief`. Offer the seed when the user has
   a specific incident in mind.

   `capabilities` defaults to the full set, so leave it alone unless the user
   describes a deliberately narrower agent. Passing `[]` leaves the agent's
   capabilities unspecified — rarely what you want, since those lines are what
   make the scenario credible to a sceptical participant.

4. **Generate from the returned prompt yourself.** `messages[0]` is the system
   framing and `messages[1]` is the task; it already carries the deployment
   archetype and its access/detection/critical-control profile, the CERT
   dimensions, the agent capabilities, the full STRIDE threat scope (IDs, names
   and descriptions), the available detection strategies, the NIST CSF control
   set, and any scenario seed and required decision points. Work from that
   payload — do not substitute your own recollection of the threat model.

   (`generate_ai_insider_scenario` does the same thing server-side but needs an
   API key. Prefer `get_ai_insider_prompt`; fall back to the generate tool only
   if the user explicitly wants AttackGen's own model to write it.)

5. **Reshape into the exercise** per `references/ai-insider-format.md`. The
   prompt's own seven sections are a scenario, not an MSEL — that reference maps
   them onto the tabletop structure and adds the inject schedule, the
   phase-grouped discussion questions and the pass criteria.

## Output format (both paths)

- **Markdown** (default) — render the sections from the path's format reference:
  `references/exercise-format.md` (Path A) or `references/ai-insider-format.md`
  (Path B).
- **HTML report** — when the user asks for a report, handout, deck,
  printable/PDF, or "make it nice," follow `references/html-report.md`: clone
  `assets/tabletop-report.html`, fill it with the real exercise, write it to the
  OS temp dir, open it, and report the path. That reference covers both paths.

`references/exercise-format.md` holds the house style (British English;
realistic for the industry and size; pace to the adversary's tradecraft) for
both paths; `references/ai-insider-format.md` carries only the Path B deltas.

## Rules

- **Data fidelity:** every fact in the exercise traces to a tool result — Path A:
  technique, ID, DET id, analytic, log source and mitigation from
  `get_kill_chain` / `get_detection_report`; Path B: STRIDE ID and name,
  archetype profile, threat category, detection strategy and NIST CSF control
  from `get_ai_insider_prompt` / `list_ai_insider_options`.
- **Recon / Resource Development** (Path A): frame these as pre-compromise
  context (mitigation `M1056`) — they carry little detection telemetry.
- **Reproducibility** (Path A): report the `seed` you passed to
  `get_kill_chain` so the user can regenerate the identical exercise. Path B has
  no seed — report the template name (or the archetype/category/STRIDE
  selections) and any `scenario_seed` text instead.
- **Rehearse the response, not the exploit** (both paths, and load-bearing on
  Path B): the deliverable is a decision-forcing exercise. Describe *what the
  adversary achieved and what the defenders must decide*, never how to reproduce
  it. No working exploit code, no payloads, no bypass steps, no real
  unpatched vulnerabilities, and no naming of real third-party organisations as
  victims — give the outside party a plausible fictional name.
- **No human motive on Path B:** an AI agent has no intent in the human sense.
  Attribute its behaviour to configuration, access, capability and objective
  (misalignment, reward hacking, emergent objectives, prompt-injection
  compromise) — never to grievance, greed or malice.
