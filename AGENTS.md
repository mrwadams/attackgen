# AGENTS.md

This file provides guidance to coding agents (e.g. OpenAI Codex) when working with code in this repository.

## Project Overview

AttackGen is a cybersecurity incident response testing tool that generates tailored attack scenarios based on the MITRE ATT&CK and ATLAS frameworks using large language models. It is built as a Streamlit web application with support for multiple LLM providers.

## Development Commands

### Running the Application
```bash
streamlit run 00_👋_Welcome.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Docker Usage
```bash
# Build image
docker build -t attackgen .

# Run container
docker run -p 8501:8501 attackgen
```

### Running Tests
```bash
# Install dev dependencies (includes pytest)
pip install -r requirements-dev.txt

# Run the suite
pytest
```
`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]`, so `pytest` from the repo root discovers everything under `tests/` with no extra flags. It also declares `[project]`/`[build-system]` metadata and the `attackgen-mcp = "mcp_server:main"` console script, so `pip install -e .` makes the package importable and installs the MCP entry point.

In CI, `.github/workflows/tests.yml` runs the same suite on Python 3.10–3.14 for every push and pull request to `main`. `release.yml` keeps its own copy of that job so a release gates on its own tag rather than on whatever last ran against `main`.

## Architecture

### Core Components
- **00_👋_Welcome.py**: Main entry point and configuration UI (sidebar is registry-driven from `core/models.py`)
- **core/**: Unified LLM wrapper. All provider integration lives here — pages never call provider SDKs directly.
  - **core/llm.py**: `call_llm(config, messages)` — the single entry point. Routes to providers via LiteLLM, applies provider-specific kwargs, wraps in LangSmith `@traceable` when a LangSmith client is available.
  - **core/models.py**: Provider + model registry (`PROVIDERS`, `MODELS`). Add or update a model by editing this file alone. `ModelInfo.supports_temperature` (default `True`) marks models that reject a custom sampling temperature; `core/llm.py:_accepts_temperature` reads it. Set it to `False` for new Anthropic models unless you have verified the model accepts one — Claude 4.7 and later return a 400.
  - **core/schemas.py**: `LLMConfig` dataclass and `LLMConfig.from_session_state(...)` factory.
  - **core/prompts.py**: Single source of truth for every scenario prompt (system prompts + the four Threat-Group/Custom human templates + the AI-insider template) and the three `build_*_messages` builders. Streamlit-free — the AI-uplift toggle is passed in as a bool via `core.ai_uplift.append_ai_uplift`, not read from session state. Pages 1–3 and the MCP server both call these builders, so there is one copy of each prompt. The template strings are lifted verbatim (including the deliberate quirk that the Custom ATTACK template alone omits the "Write in British English." line); `tests/test_prompts.py` pins them. `build_ai_insider_messages` also takes optional `scenario_seed` (free-text narrative framing) and `required_decisions` (decisions the Discussion Questions must force); both render into one `{seed_block}` in the AI-insider template that collapses to nothing when neither is supplied, so seedless prompts are byte-identical to pre-seed ones.
  - **core/attack_data.py**: Headless MITRE loaders + kill-chain resolution, shared by the pages and the MCP server. Lazy `@lru_cache` singletons (`enterprise_data`/`ics_data`/`atlas_data`) with `__file__`-derived absolute paths (so the MCP server works from any cwd and the 53 MB Enterprise bundle loads only on first use). `resolve_threat_group_kill_chain(matrix, group, *, seed=None)` reproduces the old page-1 logic exactly — dedupe for display, phase normalisation/ordering, and one-technique-per-phase sampling from the *non-deduplicated* set (pass `seed` for a deterministic draw; `None` = the UI's per-run randomness). `resolve_case_study_kill_chain` handles ATLAS (full procedure, no sampling). Returns JSON-native `KillChain` records.
  - **core/navigator.py**: Builds an ATT&CK Navigator layer JSON from a scenario's techniques, offered as a second download next to the markdown on pages 1–2. Maps each matrix to its Navigator domain (`enterprise-attack`, `ics-attack`, and the ATLAS Navigator fork's `atlas-atlas`); ATLAS layers omit the `attack` version field. Pages pass a `build_layer` callback to `run_scenario_page`; the layer is captured at generation time and persisted so it can't drift from the scenario on rerun.
  - **core/detections.py**: Purple-team "Detection & Response" companion. For a scenario's techniques it joins the defensive half of the STIX bundle already shipped — ATT&CK v18+ detection strategies + analytics (with their log sources) and mitigations — into a deterministic Markdown section (no LLM call). Enterprise/ICS resolve via `mitreattack-python` helpers; ATLAS degrades to mitigations only (it has no detection model). Pages pass a `build_defense` callback to `run_scenario_page`; the report is captured at generation time and persisted (like the layer) so it can't drift on rerun, and offered as a `..._detection.md` download. Also hosts an optional LLM "purple-team narrative" pass (a per-page `🟣 Purple-team narrative` toggle) that weaves those *supplied* detections/mitigations into a stage-by-stage defender's walkthrough — a second model call tagged `purple_team_narrative` in LangSmith.
  - **core/ai_uplift.py**: Optional "AI-enhanced adversary" framing for the Threat Group and Custom pages. A per-page toggle that appends a prompt fragment reframing the *same* kill chain as AI-accelerated (lowered skill floor, compressed timelines, autonomous orchestration) and adds an `ai_enhanced` LangSmith trace tag. Based on Anthropic's "LLM ATT&CK Navigator" research. Not used on page 3, where the AI agent is already the threat actor. The pure `append_ai_uplift(user_content, ai_uplift: bool)` is the Streamlit-free core (used by `core/prompts.py`); `apply_ai_uplift(user_content, page_id)` delegates to it after reading the toggle.
- **mcp_server.py** (repo root): MCP server (`MCPServer`, official `mcp` SDK 2.x) exposing scenario generation to agentic clients over stdio. Two tiers: **data tools** (`list_threat_groups`, `list_case_studies`, `get_kill_chain`, `get_detection_report`, `get_navigator_layer`, `list_ai_insider_options`, `get_ai_insider_prompt`) make no LLM call and need no key — they return structured MITRE data and, where useful, a ready-to-run prompt so a client's own model can generate; safe to host over HTTP. **Generate tools** (`generate_threat_group_scenario`, `generate_custom_scenario`, `generate_ai_insider_scenario`) call `core.llm.call_llm` with a bring-your-own-key `LLMConfig` (validated against `core/models.py`; `api_key` omitted → provider env var) and return finished Markdown — keep these on local stdio. The two AI-insider tools accept `template=` (a quick-start preset supplying archetype, categories, STRIDE and brief) and `scenario_seed=`; `_resolve_ai_insider_inputs` merges them so a template fills gaps and explicit arguments win. `capabilities` is the one selection no template supplies, so it defaults to the full `AGENT_CAPABILITIES` set to match page 3's multiselect — pass `[]` to opt out deliberately. Because `template` gives the other arguments defaults, `industry`/`company_size` are no longer schema-required and are enforced by a runtime guard instead. Composes `core.attack_data` + `core.prompts` + `core.detections`/`core.navigator`. Launch: `python -m mcp_server` or the `attackgen-mcp` console script.
- **skills/**: [Agent Skills](https://agentskills.io) — `SKILL.md` folders teaching a skill-aware client how to use the MCP tools well. They bundle no data (it always comes from the MCP server) and make no LLM call of their own, so they run on the caller's model with no API key. `skills/attackgen-tabletop/` turns tool output into a facilitator-ready tabletop/MSEL and has two paths: **Path A** builds the exercise from a MITRE kill chain (`get_kill_chain` + `get_detection_report`), **Path B** from the AI insider-threat model (`list_ai_insider_options` + `get_ai_insider_prompt`), where the exercise spine is the STRIDE threat scope and deployment archetype rather than ATT&CK phases. `references/` are loaded on demand: `exercise-format.md` (house style + Path A structure), `ai-insider-format.md` (Path B deltas), `html-report.md` (both paths). Changing an AI-insider template's `required_decisions` or adding a threat category changes what the skill emits, so keep `ai-insider-format.md` in step with `data/ai_insider_threats.py`.
- **pages/**: Streamlit pages for different functionality
  - **1_🛡️_Threat_Group_Scenarios.py**: Generate scenarios based on threat actor groups (ATT&CK) or case studies (ATLAS). Supports the optional AI-enhanced adversary toggle (`core/ai_uplift.py`).
  - **2_🛠️_Custom_Scenarios.py**: Generate custom scenarios from selected ATT&CK / ATLAS techniques. Supports the optional AI-enhanced adversary toggle (`core/ai_uplift.py`).
  - **3_🤖_AI_Insider_Threat_Scenarios.py**: Generate scenarios where a frontier AI agent deployed inside the organisation acts as an insider threat (based on the "Actions Speak Louder Than Tokens" threat model). Driven by deployment archetype, threat taxonomy, STRIDE threats and an optional free-text scenario seed rather than a MITRE matrix. Picking a template pre-fills the seed box from its `brief` and stores its `required_decisions` in session state (cleared when the template is deselected, since they have no widget of their own).
  - **4_💬_AttackGen_Assistant.py**: Chat interface for refining scenarios

### Data Files
- **data/enterprise-attack.json**: MITRE ATT&CK Enterprise matrix (STIX format)
- **data/ics-attack.json**: MITRE ATT&CK ICS matrix (STIX format)
- **data/stix-atlas.json**: MITRE ATLAS matrix (STIX format)
- **data/groups.json**: Threat actor group mappings for Enterprise
- **data/groups_ics.json**: Threat actor group mappings for ICS
- **data/atlas-case-studies.json**: ATLAS case studies
- **data/ai_insider_threats.py**: AI insider threat framework (deployment archetypes, threat taxonomy, STRIDE threats, CERT dimensions, detection strategies, NIST CSF controls, and templates) used by the AI Insider Threat Scenarios page. **Extends** the published paper rather than mirroring it: the paper documents 23 STRIDE threats across five categories, while this module adds a sixth category (Containment & Third-Party Impact) and four threats (`S5` reusable workload credential replay, `R4` agent-side telemetry suppression, `I5` third-party/downstream compromise, `E5` containment boundary assurance failure) covering the July 2026 disclosures of evaluation agents reaching real systems from environments assumed isolated. Templates carry two keys beyond their selections — `brief` (narrative seed injected into the prompt and pre-filled into the page's seed box) and `required_decisions` (decisions the Discussion Questions must force). These split templates into two classes: **archetype** templates (the original six) set `required_decisions` only, so the model invents the premise and re-running one yields a fresh exercise; **incident** templates (currently only Evaluation Environment Escape) also set `brief`, pinning a specific situation. Both keys are independently optional in the builder — `required_decisions` renders without a seed and vice versa. Decisions must be written `Lead — detail`; the page caption renders the lead alone. Resolve one via `resolve_template(name)`, which always returns all five keys and raises `ValueError` on an unknown name; the page and the MCP server share it. Note that `S3`, `T3`, `T4`, `I4`, `E3` and `E4` belong to no category and are reachable only from the STRIDE list directly.

### Configuration
- **.env**: API keys and secrets (not tracked in git)
- **.streamlit/secrets.toml**: Streamlit secrets for LangSmith integration
- **requirements.txt**: Python dependencies — `litellm`, `mcp`, `openai`, `streamlit`, `mitreattack-python`, `pandas`, `langsmith`, `python-dotenv`
- **requirements-dev.txt**: Development dependencies (`pytest`); installs `requirements.txt` plus test tooling

### Key Libraries
- **Streamlit**: Web application framework
- **LiteLLM**: Unified interface to all LLM providers — wrapped by `core/llm.py`
- **mitreattack-python**: MITRE ATT&CK data processing
- **pandas**: Data manipulation for technique selection
- **python-dotenv**: Environment variable management
- **langsmith**: Optional tracing (`@traceable`) and feedback logging

### LLM Provider Support
All providers go through `core/llm.py:call_llm`. Currently supported:
- OpenAI API (GPT-5.x family)
- Anthropic API (Claude 4.x family)
- Google AI API (Gemini 3.x family)
- Mistral API
- Groq API
- Custom (any OpenAI-compatible endpoint — e.g. `http://localhost:11434/v1` for Ollama, `http://localhost:1234/v1` for LM Studio, OpenRouter, an Azure OpenAI deployment, etc.)

**Adding a new model** is a one-file change: add a `ModelInfo` entry to the `MODELS` list in `core/models.py`. The sidebar picks it up automatically. Check whether the model accepts a custom sampling temperature and set `supports_temperature=False` if not.

**Adding a new provider** also lives in `core/models.py` — add a `ProviderInfo` entry with the appropriate `litellm_prefix`.

### Scenario Generation Flow
1. User selects model provider and configures API credentials (sidebar).
2. User chooses organisation details (industry, size) and MITRE framework.
3. For threat group scenarios: select threat actor group / ATLAS case study → extract techniques → generate scenario.
4. For custom scenarios: select individual ATT&CK / ATLAS techniques → generate scenario.
5. For AI insider threat scenarios: pick deployment archetype, threat categories and STRIDE threats (optionally from a template), optionally add a scenario seed → generate scenario.
6. Optional: use the AttackGen Assistant to refine generated scenarios.

### Environment Variables
Loaded from `.env`:
- Model API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `CUSTOM_API_KEY`
- Custom endpoint: `CUSTOM_BASE_URL`, `CUSTOM_MODEL_NAME`
- Optional: `LANGCHAIN_API_KEY` for LangSmith tracing (also configurable via `.streamlit/secrets.toml`)
