# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Architecture

### Core Components
- **00_👋_Welcome.py**: Main entry point and configuration UI (sidebar is registry-driven from `core/models.py`)
- **core/**: Unified LLM wrapper. All provider integration lives here — pages never call provider SDKs directly.
  - **core/llm.py**: `call_llm(config, messages)` — the single entry point. Routes to providers via LiteLLM, applies provider-specific kwargs, wraps in LangSmith `@traceable` when a LangSmith client is available.
  - **core/models.py**: Provider + model registry (`PROVIDERS`, `MODELS`). Add or update a model by editing this file alone.
  - **core/schemas.py**: `LLMConfig` dataclass and `LLMConfig.from_session_state(...)` factory.
  - **core/ai_uplift.py**: Optional "AI-enhanced adversary" framing for the Threat Group and Custom pages. A per-page toggle that appends a prompt fragment reframing the *same* kill chain as AI-accelerated (lowered skill floor, compressed timelines, autonomous orchestration) and adds an `ai_enhanced` LangSmith trace tag. Based on Anthropic's "LLM ATT&CK Navigator" research. Not used on page 3, where the AI agent is already the threat actor.
- **pages/**: Streamlit pages for different functionality
  - **1_🛡️_Threat_Group_Scenarios.py**: Generate scenarios based on threat actor groups (ATT&CK) or case studies (ATLAS). Supports the optional AI-enhanced adversary toggle (`core/ai_uplift.py`).
  - **2_🛠️_Custom_Scenarios.py**: Generate custom scenarios from selected ATT&CK / ATLAS techniques. Supports the optional AI-enhanced adversary toggle (`core/ai_uplift.py`).
  - **3_🤖_AI_Insider_Threat_Scenarios.py**: Generate scenarios where a frontier AI agent deployed inside the organisation acts as an insider threat (based on the "Actions Speak Louder Than Tokens" threat model). Driven by deployment archetype, threat taxonomy, and STRIDE threats rather than a MITRE matrix.
  - **4_💬_AttackGen_Assistant.py**: Chat interface for refining scenarios

### Data Files
- **data/enterprise-attack.json**: MITRE ATT&CK Enterprise matrix (STIX format)
- **data/ics-attack.json**: MITRE ATT&CK ICS matrix (STIX format)
- **data/stix-atlas.json**: MITRE ATLAS matrix (STIX format)
- **data/groups.json**: Threat actor group mappings for Enterprise
- **data/groups_ics.json**: Threat actor group mappings for ICS
- **data/atlas-case-studies.json**: ATLAS case studies
- **data/ai_insider_threats.py**: AI insider threat framework (deployment archetypes, threat taxonomy, STRIDE threats, CERT dimensions, detection strategies, NIST CSF controls, and templates) used by the AI Insider Threat Scenarios page

### Configuration
- **.env**: API keys and secrets (not tracked in git)
- **.streamlit/secrets.toml**: Streamlit secrets for LangSmith integration
- **requirements.txt**: Python dependencies — `litellm`, `streamlit`, `mitreattack-python`, `pandas`, `langsmith`

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

**Adding a new model** is a one-file change: add a `ModelInfo` entry to the `MODELS` list in `core/models.py`. The sidebar picks it up automatically.

**Adding a new provider** also lives in `core/models.py` — add a `ProviderInfo` entry with the appropriate `litellm_prefix`.

### Scenario Generation Flow
1. User selects model provider and configures API credentials (sidebar).
2. User chooses organisation details (industry, size) and MITRE framework.
3. For threat group scenarios: select threat actor group / ATLAS case study → extract techniques → generate scenario.
4. For custom scenarios: select individual ATT&CK / ATLAS techniques → generate scenario.
5. For AI insider threat scenarios: pick deployment archetype, threat categories and STRIDE threats → generate scenario.
6. Optional: use the AttackGen Assistant to refine generated scenarios.

### Environment Variables
Loaded from `.env`:
- Model API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `CUSTOM_API_KEY`
- Custom endpoint: `CUSTOM_BASE_URL`, `CUSTOM_MODEL_NAME`
- Optional: `LANGCHAIN_API_KEY` for LangSmith tracing (also configurable via `.streamlit/secrets.toml`)
