# HTML report mode

When the user asks for an **HTML report / deck / handout / something to run the
exercise from** (or just says "make it nice"), produce a self-contained HTML
report in place of the Markdown default. Same content and data rules as the
path's format reference (`references/exercise-format.md` for Path A,
`references/ai-insider-format.md` for Path B) — this file only covers the
delivery mechanics, and applies to both.

## Produce the file

1. **Build the exercise data first** via the AttackGen MCP tools exactly as in
   `SKILL.md` — `get_kill_chain` + `get_detection_report` on Path A,
   `get_ai_insider_prompt` on Path B. Have the real techniques and detection
   data (or the real archetype and STRIDE payload) in hand before writing any
   HTML.

2. **Clone the bundled template** `assets/tabletop-report.html` — reproduce its
   structure and inline CSS, and replace the example content (APT29 / Northgate)
   with the real exercise. The template is the design system; keep its look.

3. **Write it to the OS temp directory** (keeps the user's repo clean). Resolve
   the temp dir from `$TMPDIR`, falling back to `/tmp` (or `%TEMP%` on Windows),
   and use a timestamped name:

   ```
   <tmpdir>/attackgen-tabletop-<subject>-<timestamp>.html
   ```

   `<subject>` is the threat group / case study on Path A, or the template name
   (falling back to the archetype) on Path B. Sanitise it to a filename-safe
   slug.

4. **Open it and report the path.** `open <path>` on macOS, `xdg-open <path>` on
   Linux, `start <path>` on Windows. Then tell the user the absolute file path.

## Hard requirements

- **Fully self-contained.** Embed everything in the one file — inline CSS and JS,
  system-font stacks, and hand-built CSS/SVG for visuals — so the handout renders
  offline, prints to a clean PDF, emails cleanly, and opens safely in an
  air-gapped SOC. (A deliberate departure from CDN-based report skills; keep it
  network-free.)
- **Print-friendly.** Keep the template's `@media print` block: it hides the
  toolbar, expands collapsed sections, reveals facilitator material, and keeps
  cards intact across page breaks — so "Print / PDF" yields a usable handout.
- **Theme-aware.** Keep the `prefers-color-scheme` light/dark variables.
- **Data fidelity.** As in `SKILL.md`, every technique, DET id, log source,
  mitigation, STRIDE ID and archetype field in the HTML traces to an MCP tool
  output.

## Interactive touches to preserve

- **Facilitator ⇄ participant toggle.** Mark the kill-chain table and the
  detection scorecard `class="facilitator-only"`; the toolbar toggle hides them so
  the same file can be shown to participants and then revealed at the hotwash.
- **Inject timeline.** Render section 3 as the CSS timeline; give the inject the
  IR team starts from `class="start"` and a start tag.
- **Detection scorecard.** Render the `get_detection_report` data as the scorecard
  table with a Yes/Partial/No control per technique, so the facilitator scores
  detection coverage live.

## Path B differences

The template is Path A-shaped (its example content is APT29 / Northgate), but the
six-section spine is the same. Reuse it with these substitutions — keep the
markup, CSS, classes and print rules exactly as they are:

- **Section 2** becomes *Agent Profile & Threat Scope*. Keep the
  `facilitator-only` wrapper and the `Facilitator` flag. Put the archetype block
  above the table, and change the table's columns to
  `# | STRIDE (ID) | Threat | Scenario action`.
- **Section 5** becomes the *Detection Coverage Scorecard*, one row per STRIDE
  threat rather than per technique, with the same Yes/Partial/No control. Add
  the archetype's detection posture as a lead-in line, so the facilitator can see
  why a row is expected to fail.
- **The header metadata block** takes Framework / Deployment archetype / Threat
  scope / Target profile / Exercise type, in place of the threat-actor fields.
- **Section 3** is unchanged mechanically: tag each inject with its STRIDE ID
  instead of an ATT&CK ID, and put `class="start"` on the first
  *defender-visible* signal — often an external one on this path.

## Optional enhancements (only if the user wants them)

- A coverage tally that counts scorecard selections (small inline JS).
- A hand-built SVG kill-chain ribbon across the top — inline SVG, hand-authored,
  no library needed.
