# HTML report mode

When the user asks for an **HTML report / deck / handout / something to run the
exercise from** (or just says "make it nice"), produce a self-contained HTML
report in place of the Markdown default. Same content and data rules as
`references/exercise-format.md` — this file only covers the delivery mechanics.

## Produce the file

1. **Build the exercise data first** via the AttackGen MCP tools (`get_kill_chain`,
   `get_detection_report`) exactly as in `SKILL.md`. Have the real techniques and
   detection data in hand before writing any HTML.

2. **Clone the bundled template** `assets/tabletop-report.html` — reproduce its
   structure and inline CSS, and replace the example content (APT29 / Northgate)
   with the real exercise. The template is the design system; keep its look.

3. **Write it to the OS temp directory** (keeps the user's repo clean). Resolve
   the temp dir from `$TMPDIR`, falling back to `/tmp` (or `%TEMP%` on Windows),
   and use a timestamped name:

   ```
   <tmpdir>/attackgen-tabletop-<group>-<timestamp>.html
   ```

   Sanitise `<group>` to a filename-safe slug.

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
- **Data fidelity.** As in `SKILL.md`, every technique, DET id, log source, and
  mitigation in the HTML traces to an MCP tool output.

## Interactive touches to preserve

- **Facilitator ⇄ participant toggle.** Mark the kill-chain table and the
  detection scorecard `class="facilitator-only"`; the toolbar toggle hides them so
  the same file can be shown to participants and then revealed at the hotwash.
- **Inject timeline.** Render section 3 as the CSS timeline; give the inject the
  IR team starts from `class="start"` and a start tag.
- **Detection scorecard.** Render the `get_detection_report` data as the scorecard
  table with a Yes/Partial/No control per technique, so the facilitator scores
  detection coverage live.

## Optional enhancements (only if the user wants them)

- A coverage tally that counts scorecard selections (small inline JS).
- A hand-built SVG kill-chain ribbon across the top — inline SVG, hand-authored,
  no library needed.
