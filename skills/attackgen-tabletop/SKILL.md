---
name: attackgen-tabletop
description: >-
  Turn a MITRE ATT&CK/ATLAS threat group or case study into a full
  incident-response tabletop exercise (MSEL) — scenario narrative, kill chain,
  timestamped injects, discussion questions, and a detection-coverage scorecard —
  using the AttackGen MCP server. Use when the user wants a tabletop, MSEL, or
  purple-team IR exercise for a specific threat actor, ICS group, ATLAS case
  study, or set of ATT&CK/ATLAS techniques, or asks for one as a printable HTML
  report.
version: 1.0.0
license: GPL-3.0-or-later
---

# AttackGen Tabletop / MSEL

Turn AttackGen's MITRE data into a facilitator-ready tabletop exercise. This
skill orchestrates the AttackGen MCP tools and formats their output; the attack
and detection data always comes live from those tools.

## Prerequisites

The **AttackGen MCP server** must be connected. You will use (tool names may be
namespaced by your client, e.g. `attackgen.get_kill_chain`):

- `list_threat_groups(matrix)` / `list_case_studies()` — resolve a group or
  case-study name.
- `get_kill_chain(matrix, group, seed?, industry?, company_size?)` — the kill
  chain (one technique per phase for ATT&CK; full procedure for ATLAS).
- `get_detection_report(matrix, technique_ids)` — the Detection & Response join
  (ATT&CK v18 detection strategies + analytics + mitigations).
- `get_navigator_layer(matrix, technique_ids, name?, description?)` — optional
  Navigator layer JSON for a coverage visual.

If these tools aren't connected, ask the user to add the server
(`claude mcp add attackgen -- python -m mcp_server`, or their client's
equivalent) and stop there.

## Workflow

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

5. **Choose the format:**
   - **Markdown** (default) — render the sections in `references/exercise-format.md`.
   - **HTML report** — when the user asks for a report, handout, deck,
     printable/PDF, or "make it nice," follow `references/html-report.md`: clone
     `assets/tabletop-report.html`, fill it with the real exercise, write it to
     the OS temp dir, open it, and report the path.

6. **Author the exercise** in the chosen format, tying every inject and
   discussion question to the specific techniques from step 2 and the detection
   data from step 3. `references/exercise-format.md` defines the section
   structure and house style (British English; realistic for the industry and
   size; pace to the actor's tradecraft) for both formats.

## Rules

- **Data fidelity:** every technique, ID, DET id, analytic, log source, and
  mitigation in the exercise traces to a `get_kill_chain` or
  `get_detection_report` result.
- **Recon / Resource Development:** frame these as pre-compromise context
  (mitigation `M1056`) — they carry little detection telemetry.
- **Reproducibility:** report the `seed` you passed to `get_kill_chain` so the
  user can regenerate the identical exercise.
