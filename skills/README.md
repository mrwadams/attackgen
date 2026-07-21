# AttackGen Agent Skills

[Agent Skills](https://agentskills.io) are an open, cross-vendor format
(`SKILL.md` folders) for teaching an agent *how* to do a task well. They pair
with the [AttackGen MCP server](../README.md#mcp-server): the **MCP tools supply
the live MITRE data** (kill chains + the ATT&CK v18 detection join), and a
**skill supplies the craft** — the workflow and output format AttackGen uses to
turn that data into a finished exercise. The skill runs on the caller's own
model, so it needs no API key.

## Skills in this repo

| Skill | What it does |
|-------|--------------|
| [`attackgen-tabletop/`](attackgen-tabletop/SKILL.md) | Turns a threat group / ATLAS case study into a full incident-response **tabletop / MSEL** — narrative, phase-mapped kill chain, timestamped inject schedule, NIST 800-61 discussion questions, a detection-coverage reference, and pass criteria. Orchestrates the AttackGen MCP data tools. |

Each skill is a folder with a `SKILL.md` (metadata + instructions) and optional
`references/` loaded on demand via progressive disclosure — no data is bundled,
so the MITRE content never goes stale (it always comes from the MCP server).

## Installing

Skills are supported by a broad set of agents (Claude Code, OpenAI Codex, Gemini
CLI, Cursor, GitHub Copilot / VS Code, Goose, OpenCode, and more — see the
[client showcase](https://agentskills.io/clients)). Each client has its own
skills directory; consult its docs. Common patterns:

- **Claude Code:** copy the skill folder into `.claude/skills/` (project) or
  `~/.claude/skills/` (personal), or ship it in a plugin (below).
- **Other clients:** place the skill folder in that client's skills directory
  (e.g. `.codex/skills/`, `.gemini/skills/`, `.cursor/skills/`).

The skills assume the **AttackGen MCP server is connected** (see the main
README's *MCP Server* section). Without it they will ask you to add it rather
than fabricate data.

## Bundling skills + the MCP server as one install (Claude Code plugin)

A Claude Code plugin can ship the MCP server config **and** the skills together,
so "add AttackGen to your agent" is a single step. Sketch:

```
attackgen-plugin/
├── .claude-plugin/
│   └── plugin.json
├── .mcp.json                     # registers the attackgen MCP server
└── skills/
    └── attackgen-tabletop/
        ├── SKILL.md
        └── references/exercise-format.md
```

`.mcp.json` (the plugin's MCP registration):

```json
{
  "mcpServers": {
    "attackgen": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/absolute/path/to/attackgen"
    }
  }
}
```

Installing the plugin then gives the user both the tools (capability) and the
skills (craft) at once. For non-plugin clients, install the MCP server and copy
the skill folder into the client's skills directory separately.
