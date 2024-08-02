# AttackGen

AttackGen is a cybersecurity incident response testing tool that leverages the power of large language models and the comprehensive MITRE ATT&CK framework. The tool generates tailored incident response scenarios based on user-selected threat actor groups and your organisation's details.

## Table of Contents
- [Star the Repo](#star-the-repo)
- [Features](#features)
- [Releases](#releases)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running AttackGen](#running-attackgen)
- [Usage](#usage)
- [Contributing](#contributing)
- [Licence](#licence)

## Star the Repo
If you find AttackGen useful, please consider starring the repository on GitHub. This helps more people discover the tool. Your support is greatly appreciated! ⭐


## Features

- Generates unique incident response scenarios based on chosen threat actor groups.
- Allows you to specify your organisation's size and industry for a tailored scenario.
- Displays a detailed list of techniques used by the selected threat actor group as per the MITRE ATT&CK framework.
- Create custom scenarios based on a selection of ATT&CK techniques.
- 🆕 Use scenario templates to quickly generate custom scenarios based on common types of cyber incidents.
- 🆕 AttackGen Assistant - a chat interface for updating and/or asking questions about generated scenarios.
- Capture user feedback on the quality of the generated scenarios.
- Downloadable scenarios in Markdown format.
- Use the OpenAI API, Azure OpenAI Service, 🆕 Google AI API, Mistral API, or locally hosted Ollama models to generate incident response scenarios.
- Available as a Docker container image for easy deployment.
- Optional integration with [LangSmith](https://docs.smith.langchain.com/) for powerful debugging, testing, and monitoring of model performance.

![AttackGen Screenshot](./images/screenshot.png)

## Releases

### v0.5.2
| What's new? | Why is it useful? |
| ----------- | ---------------- |
| GPT-4o mini Model Support | - Expanded Model Options: AttackGen now supports the use of OpenAI's GPT-4o mini model. GPT4-o mini is a highly cost-efficient small model. |
| Updated model choices for Google AI API | - Expanded Model Options: AttackGen now supports the stable and preview versions of Gemini 1.5 Pro. | 

### v0.5.1
| What's new? | Why is it useful? |
| ----------- | ---------------- |
| GPT-4o Model Support | - Enhanced Model Options: AttackGen now supports the use of OpenAI's GPT-4o model. GPT4-o is OpenAI's leading model, able to generate scenarios twice as fast as GPT-4 for half the cost. |

### v0.5
| What's new? | Why is it useful? |
| ----------- | ---------------- |
| AttackGen Assistant | - Iterative Scenario Refinement: The new chat interface allows users to interact with their generated incident response scenarios, making it easy to update and ask questions about the scenario without having to regenerate it from scratch. This feature enables an iterative approach to scenario development, where users can refine and improve their scenarios based on the AI assistant's responses.<br><br>- Contextual Assistance: The AI assistant responds to user queries based on the context of the generated scenario and the conversation history. This ensures that the assistant's responses are relevant and helpful in refining the scenario. |
| Quick Start Templates for Custom Scenarios | - Quick Scenario Generation: Users can now quickly generate custom incident response scenarios based on predefined templates for common types of cyber incidents, such as phishing attacks, ransomware attacks, malware infections, and insider threats. This feature makes it easier to create realistic scenarios without having to select individual ATT&CK techniques.<br><br>- Streamlined Workflow: The template selection is integrated seamlessly into the custom scenario generation process. Users can choose a template, which automatically populates the relevant ATT&CK techniques, and then further customize the scenario if needed. |
| Google AI API Integration  | - Expanded Model Options: AttackGen now supports the use of Google's Gemini models for generating incident response scenarios. This integration expands the range of high-quality models available to users, allowing them to leverage Google's AI capabilities for creating realistic and diverse scenarios. |

<details>
  <summary>Click to view release notes for earlier versions.</summary>

### v0.4

| What's new? | Why is it useful? |
| ----------- | ---------------- |
| Mistral API Integration | - Alternative Model Provider: Users can now leverage the Mistral AI models to generate incident response scenarios. This integration provides an alternative to the OpenAI and Azure OpenAI Service models, allowing users to explore and compare the performance of different language models for their specific use case. |
| Local Model Support using Ollama | - Local Model Hosting: AttackGen now supports the use of locally hosted LLMs via an integration with Ollama. This feature is particularly useful for organisations with strict data privacy requirements or those who prefer to keep their data on-premises. Please note that this feature is not available for users of the AttackGen version hosted on Streamlit Community Cloud at https://attackgen.streamlit.app |
| Optional LangSmith Integration | - Improved Flexibility: The integration with LangSmith is now optional. If no LangChain API key is provided, users will see an informative message indicating that the run won't be logged by LangSmith, rather than an error being thrown. This change improves the overall user experience and allows users to continue using AttackGen without the need for LangSmith. |
| Various Bug Fixes and Improvements | - Enhanced User Experience: This release includes several bug fixes and improvements to the user interface, making AttackGen more user-friendly and robust. |

### v0.3

| What's new? | Why is it useful? |
| ----------- | ---------------- |
| Azure OpenAI Service Integration | - Enhanced Integration: Users can now choose to utilise OpenAI models deployed on the Azure OpenAI Service, in addition to the standard OpenAI API. This integration offers a seamless and secure solution for incorporating AttackGen into existing Azure ecosystems, leveraging established commercial and confidentiality agreements.<br><br>- Improved Data Security: Running AttackGen from Azure ensures that application descriptions and other data remain within the Azure environment, making it ideal for organizations that handle sensitive data in their threat models. |
| LangSmith for Azure OpenAI Service | - Enhanced Debugging: LangSmith tracing is now available for scenarios generated using the Azure OpenAI Service. This feature provides a powerful tool for debugging, testing, and monitoring of model performance, allowing users to gain insights into the model's decision-making process and identify potential issues with the generated scenarios.<br><br>- User Feedback: LangSmith also captures user feedback on the quality of scenarios generated using the Azure OpenAI Service, providing valuable insights into model performance and user satisfaction. |
| Model Selection for OpenAI API | - Flexible Model Options: Users can now select from several models available from the OpenAI API endpoint, such as `gpt-4-turbo-preview`. This allows for greater customization and experimentation with different language models, enabling users to find the most suitable model for their specific use case. |
| Docker Container Image | - Easy Deployment: AttackGen is now available as a Docker container image, making it easier to deploy and run the application in a consistent and reproducible environment. This feature is particularly useful for users who want to run AttackGen in a containerised environment, or for those who want to deploy the application on a cloud platform. |



### v0.2

| What's new? | Why is it useful? |
| ----------- | ---------------- |
| Custom Scenarios based on ATT&CK Techniques | - For Mature Organisations: This feature is particularly beneficial if your organisation has advanced threat intelligence capabilities. For instance, if you're monitoring a newly identified or lesser-known threat actor group, you can tailor incident response testing scenarios specific to the techniques used by that group.<br><br>- Focused Testing: Alternatively, use this feature to focus your incident response testing on specific parts of the cyber kill chain or certain MITRE ATT&CK Tactics like 'Lateral Movement' or 'Exfiltration'. This is useful for organisations looking to evaluate and improve specific areas of their defence posture. |
| User feedback on generated scenarios | - Collecting feedback is essential to track model performance over time and helps to highlight strengths and weaknesses in scenario generation tasks. |
| Improved error handling for missing API keys | - Improved user experience. |
| Replaced Streamlit `st.spinner` widgets with new `st.status` widget | - Provides better visibility into long running processes (i.e. scenario generation). |

### v0.1

Initial release.

</details>

## Requirements

- Recent version of Python.
- Python packages: pandas, streamlit, and any other packages necessary for the custom libraries (`langchain` and `mitreattack`).
- OpenAI API key.
- LangChain API key (optional) - see [LangSmith Setup](#langsmith-setup) section below for further details.
- Data files: `enterprise-attack.json` (MITRE ATT&CK dataset in STIX format) and `groups.json`.

## Installation

### Option 1: Cloning the Repository

1. Clone this repository:
```
git clone https://github.com/mrwadams/attackgen.git
```

2. Change directory into the cloned repository:

```
cd attackgen
```

3. Install the required Python packages:

```
pip install -r requirements.txt
```

### Option 2: Using Docker

1. Pull the Docker container image from Docker Hub:

```
docker pull mrwadams/attackgen
```

## LangSmith Setup

If you would like to use LangSmith for debugging, testing, and monitoring of model performance, you will need to set up a LangSmith account and create a `.streamlit/secrets.toml` file that contains your LangChain API key. Please follow the instructions [here](https://docs.smith.langchain.com/) to set up your account and obtain your API key. You'll find a `secrets.toml-example` file in the `.streamlit/` directory that you can use as a template for your own secrets.toml file.

If you do not wish to use LangSmith, you must still have a `.streamlit/secrets.toml` file in place, but you can leave the `LANGCHAIN_API_KEY` field empty.

## Data Setup

Download the latest version of the MITRE ATT&CK dataset in STIX format from [here](https://github.com/mitre-attack/attack-stix-data/blob/master/enterprise-attack/enterprise-attack.json). Ensure to place this file in the `./data/` directory within the repository.

## Running AttackGen

After the data setup, you can run AttackGen with the following command:

```
streamlit run 👋_Welcome.py
```

You can also try the app on [Streamlit Community Cloud](https://attackgen.streamlit.app/).

## Usage

### Running AttackGen

#### Option 1: Running the Streamlit App Locally
1. Run the Streamlit app:
```
streamlit run 👋_Welcome.py
```
2. Open your web browser and navigate to the URL provided by Streamlit.
3. Use the app to generate standard or custom incident response scenarios (see below for details).

#### Option 2: Using the Docker Container Image
1. Run the Docker container:
```
docker run -p 8501:8501 mrwadams/attackgen
```
This command will start the container and map port 8501 (default for Streamlit apps) from the container to your host machine.
2. Open your web browser and navigate to `http://localhost:8501`.
3. Use the app to generate standard or custom incident response scenarios (see below for details).

### Generating Scenarios

#### Standard Scenario Generation
1. Choose whether to use the OpenAI API or the Azure OpenAI Service.
2. Enter your OpenAI API key, or the API key and deployment details for your model on the Azure OpenAI Service.
3. Select your organisatin's industry and size from the dropdown menus.
4. Navigate to the `Threat Group Scenarios` page.
5. Select the Threat Actor Group that you want to simulate.
6. Click on 'Generate Scenario' to create the incident response scenario.
7. Use the 👍 or 👎 buttons to provide feedback on the quality of the generated scenario. N.B. The feedback buttons only appear if a value for LANGCHAIN_API_KEY has been set in the `.streamlit/secrets.toml` file.

#### Custom Scenario Generation
1. Choose whether to use the OpenAI API or the Azure OpenAI Service.
2. Enter your OpenAI API Key, or the API key and deployment details for your model on the Azure OpenAI Service.
3. Select your organisation's industry and size from the dropdown menus.
4. Navigate to the `Custom Scenario` page.
5. Use the multi-select box to search for and select the ATT&CK techniques relevant to your scenario.
6. Click 'Generate Scenario' to create your custom incident response testing scenario based on the selected techniques.
7. Use the 👍 or 👎 buttons to provide feedback on the quality of the generated scenario. N.B. The feedback buttons only appear if a value for LANGCHAIN_API_KEY has been set in the `.streamlit/secrets.toml` file.

Please note that generating scenarios may take a minute or so. Once the scenario is generated, you can view it on the app and also download it as a Markdown file.

## Contributing

I'm very happy to accept contributions to this project. Please feel free to submit an issue or pull request.

## Licence

This project is licensed under [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
