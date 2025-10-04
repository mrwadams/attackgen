# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AttackGen is a cybersecurity incident response testing tool that generates tailored attack scenarios based on MITRE ATT&CK framework and threat actor groups using large language models. It's built as a Streamlit web application with support for multiple LLM providers.

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
- **00_👋_Welcome.py**: Main entry point and configuration UI
- **pages/**: Streamlit pages for different functionality
  - **1_🛡️_Threat_Group_Scenarios.py**: Generate scenarios based on threat actor groups
  - **2_🛠️_Custom_Scenarios.py**: Generate custom scenarios from selected ATT&CK techniques  
  - **3_💬_AttackGen_Assistant.py**: Chat interface for refining scenarios

### Data Files
- **data/enterprise-attack.json**: MITRE ATT&CK Enterprise matrix (STIX format)
- **data/ics-attack.json**: MITRE ATT&CK ICS matrix (STIX format)
- **data/groups.json**: Threat actor group mappings for Enterprise
- **data/groups_ics.json**: Threat actor group mappings for ICS

### Configuration
- **.env**: API keys and secrets (not tracked in git)
- **.streamlit/secrets.toml**: Streamlit secrets for LangSmith integration
- **requirements.txt**: Python dependencies including langchain, streamlit, pandas, mitreattack-python

### Key Libraries
- **Streamlit**: Web application framework
- **LangChain**: LLM integration and prompt management  
- **mitreattack-python**: MITRE ATT&CK data processing
- **pandas**: Data manipulation for technique selection
- **python-dotenv**: Environment variable management

### LLM Provider Support
The application supports multiple model providers through a unified interface:
- OpenAI API (using new Responses API)
- Anthropic API (Claude models)
- Azure OpenAI Service
- Google AI API (Gemini models)
- Mistral API
- Groq API  
- Ollama (local models)
- Custom OpenAI-compatible endpoints

### OpenAI Responses API Integration
AttackGen now uses OpenAI's latest Responses API for all supported models, providing:
- Unified interface across all OpenAI models (GPT-5, GPT-4.1, GPT-4o, o-series)
- Improved response formatting with proper markdown rendering
- Simplified code architecture without model-specific handling
- Enhanced structured output parsing from the API

### Scenario Generation Flow
1. User selects model provider and configures API credentials
2. User chooses organization details (industry, size) and ATT&CK matrix
3. For threat group scenarios: Select threat actor group → Extract techniques → Generate scenario
4. For custom scenarios: Select individual ATT&CK techniques → Generate scenario
5. Optional: Use AttackGen Assistant to refine generated scenarios

### Environment Variables
Key environment variables loaded from .env:
- Model API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.
- Azure-specific: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_DEPLOYMENT`
- Optional: `LANGCHAIN_API_KEY` for LangSmith tracing