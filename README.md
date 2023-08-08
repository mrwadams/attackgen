# AttackGen

AttackGen is a cybersecurity incident response testing tool that leverages the power of large language models and the comprehensive MITRE ATT&CK framework. The tool generates tailored incident response scenarios based on user-selected threat actor groups and your organization's details.

## Features

- Generates unique incident response scenarios based on chosen threat actor groups.
- Allows you to specify your organisation's size and industry for a tailored scenario.
- Displays a detailed list of techniques used by the selected threat actor group as per the MITRE ATT&CK framework.
- Downloadable scenarios in Markdown format.

## Requirements

- Recent version of Python.
- Python packages: pandas, streamlit, and any other packages necessary for the custom libraries (`langchain` and `mitreattack`).
- OpenAI API key.
- Data files: `enterprise-attack.json` (MITRE ATT&CK dataset in STIX format) and `groups.json`.

## Installation

1. Clone the repository:

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

## Data Setup

Download the latest version of the MITRE ATT&CK dataset in STIX format from [here](https://github.com/mitre-attack/attack-stix-data/blob/master/enterprise-attack/enterprise-attack.json). Ensure to place this file in the `./data/` directory within the repository.

## Running AttackGen

After the data setup, you can run AttackGen with the following command:

```
streamlit run app.py
```

## Usage

1. Enter your OpenAI API Key.
2. Select your organisation's industry and size from the dropdown menus.
3. Select a Threat Actor Group that you want to simulate.
4. Click on 'Generate Scenario' to create the incident response scenario.

Please note that generating a scenario may take a minute or so. Once the scenario is generated, you can view it on the app and also download it as a Markdown file.

## Contributing

I'm very happy to accept contributions to this project. Please feel free to submit an issue or pull request.

## License

This project is licensed under [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/).
