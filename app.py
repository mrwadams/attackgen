"""
Copyright (C) 2023, Matthew Adams

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or 
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the licence is provided with this program. If you are unable
to view it, please see https://www.gnu.org/licenses/
"""

import pandas as pd
import streamlit as st
import os

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from mitreattack.stix20 import MitreAttackData

# Add environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Define a class for handling Streamlit's callback operations
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    # Define a method for handling new tokens from the Language Model
    def on_llm_new_token(self, token: str, role: str, **kwargs) -> None:
        if role == 'assistant':
            self.text += token
            self.container.markdown(self.text)

def generate_scenario(openai_api_key, messages):
    with st.spinner('Generating scenario...'):
        llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=False)
        response = llm(messages)
    return response

with st.sidebar:
    st.sidebar.markdown("### <span style='color: #1DB954;'>How to use AttackGen</span>", unsafe_allow_html=True)
    st.sidebar.markdown('''
    1. Enter your OpenAI API Key
    2. Select your company's industry and size
    3. Select a threat actor group
    4. Generate an incident response scenario
    ''')

    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # Add the drop-down selectors for Industry and Company Size
    industry = st.selectbox(
    "Select your company's industry:",
    sorted(['Aerospace / Defense', 'Agriculture / Food Services', 
            'Automotive', 'Construction', 'Education', 
            'Energy / Utilities', 'Finance / Banking', 
            'Government / Public Sector', 'Healthcare', 
            'Hospitality / Tourism', 'Insurance', 
            'Legal Services', 'Manufacturing', 
            'Media / Entertainment', 'Non-profit', 
            'Real Estate', 'Retail / E-commerce', 
            'Technology / IT', 'Telecommunication', 
            'Transportation / Logistics'])
    , placeholder="Select Industry")

    company_size = st.selectbox("Select your company's size:", ['Small (1-50 employees)', 'Medium (51-200 employees)', 'Large (201-1,000 employees)', 'Enterprise (1,001-10,000 employees)', 'Large Enterprise (10,000+ employees)'], placeholder="Select Company Size")
    
    st.sidebar.markdown('---')

    st.sidebar.markdown("### <span style='color: #1DB954;'>About</span>", unsafe_allow_html=True)
    st.sidebar.markdown("Welcome to AttackGen! This tool uses the MITRE ATT&CK framework and large language models to generate attack scenarios for incident response testing.") 
    st.sidebar.markdown("The MITRE ATT&CK framework is a powerful tool for understanding the tactics, techniques, and procedures (TTPs) used by threat actors; however, it can be difficult to translate this information into realistic scenarios for testing.")
    st.sidebar.markdown("AttackGen solves this problem by using large language models to quickly generate attack scenarios based on a selection of a threat actor group's known techniques.")
    st.sidebar.markdown("Created by [Matt Adams](https://www.linkedin.com/in/matthewrwadams)")

# End of sidebar

attack_data = MitreAttackData("./data/enterprise-attack.json")

groups = pd.read_json("./data/groups.json")


st.markdown("# <span style='color: #1DB954;'>AttackGen 👾</span>", unsafe_allow_html=True)
st.markdown("<span style='color: #1DB954;'> **Use MITRE ATT&CK and Large Language Models to generate attack scenarios for incident response testing.**</span>", unsafe_allow_html=True)
st.markdown("---")

selected_group_alias = st.selectbox("Select a threat actor group for the scenario",
                                     sorted(groups['group'].unique()),placeholder="Select Group", index=17) # Set APT41 as the default group as the default group has no Enterprise ATT&CK techniques

phase_name_order = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence', 
                    'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement', 
                    'Collection', 'Command and Control', 'Exfiltration', 'Impact']

phase_name_category = pd.CategoricalDtype(categories=phase_name_order, ordered=True)



try:
    # Define techniques_df as an empty dataframe
    techniques_df = pd.DataFrame()

    # define selected_techniques_df as an empty dataframe before the if condition
    selected_techniques_df = pd.DataFrame()

    if selected_group_alias != "Select Group":
        # Get the group by the selected alias
        group = attack_data.get_groups_by_alias(selected_group_alias)
        group_url = groups[groups['group'] == selected_group_alias]['url'].values[0]

        # Display the URL as a clickable link
        st.markdown( f"[View {selected_group_alias}'s page on attack.mitre.org]({group_url})")

        # Check if the group was found
        if group:
            # Get the STIX ID of the group
            group_stix_id = group[0].id

            # Get all techniques used by the group
            techniques = attack_data.get_techniques_used_by_group(group_stix_id)

            # Check if there are any techniques for the group
            if not techniques:
                st.info(f"There are no Enterprise ATT&CK techniques associated with the threat group: {selected_group_alias}")
            else:
                # Update techniques_df with the techniques
                techniques_df = pd.DataFrame(techniques)

            # Create a copy of the DataFrame for generating the LLM prompt
            techniques_df_llm = techniques_df.copy()

            # Add a 'Technique Name' column to both DataFrames
            techniques_df['Technique Name'] = techniques_df_llm['Technique Name'] = techniques_df['object'].apply(lambda x: x['name'])

            # Add a 'ATT&CK ID' column to both DataFrames
            techniques_df['ATT&CK ID'] = techniques_df_llm['ATT&CK ID'] = techniques_df['object'].apply(lambda x: attack_data.get_attack_id(x['id']))

            # Add a 'Phase Name' column to both DataFrames
            techniques_df['Phase Name'] = techniques_df_llm['Phase Name'] = techniques_df['object'].apply(lambda x: x['kill_chain_phases'][0]['phase_name'])

            # Drop duplicate techniques based on Phase Name, Technique Name and ATT&CK ID
            techniques_df = techniques_df.drop_duplicates(['Phase Name', 'Technique Name', 'ATT&CK ID'])

            # Convert the 'Phase Name' to title case and replace hyphens with spaces
            techniques_df['Phase Name'] = techniques_df['Phase Name'].str.replace('-', ' ').str.title()
            techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].str.replace('-', ' ').str.title()

            # Replace 'Command And Control' with 'Command and Control'
            techniques_df['Phase Name'] = techniques_df['Phase Name'].replace('Command And Control', 'Command and Control')
            techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].replace('Command And Control', 'Command and Control')

            # Convert the 'Phase Name' column to the ordered category in both DataFrames
            techniques_df['Phase Name'] = techniques_df['Phase Name'].astype(phase_name_category)
            techniques_df_llm['Phase Name'] = techniques_df_llm['Phase Name'].astype(phase_name_category)

            # Sort the DataFrame by the 'Phase Name' column
            techniques_df = techniques_df.sort_values('Phase Name')
            techniques_df_llm = techniques_df_llm.sort_values('Phase Name')

            # Group by 'Phase Name' and randomly select one technique from each group
            # Filter the groups to include only those that have at least one row in the LLM DataFrame
            selected_techniques_df = (techniques_df_llm.groupby('Phase Name')
                                        .apply(lambda x: x.sample(n=1) if len(x) > 0 else None)
                                        .reset_index(drop=True))

            # Sort the DataFrame by the 'Phase Name' column
            techniques_df = techniques_df.sort_values('Phase Name')

            # Select only the 'Technique Name', 'ATT&CK ID', and 'Phase Name' columns
            techniques_df = techniques_df[['Technique Name', 'ATT&CK ID', 'Phase Name']]

        if not techniques_df.empty:
            # Create an expander for the techniques
            with st.expander("Associated ATT&CK Techniques"):
                # Use the st.table function to display the DataFrame
                st.dataframe(data=techniques_df, height=200, use_container_width=True, hide_index=True)

        # Create an empty list to hold the kill chain information
        kill_chain = []

        # Iterate over the rows in the DataFrame
        for index, row in selected_techniques_df.iterrows():
            # Extract the phase and technique information
            phase_name = row['Phase Name']
            technique_name = row['Technique Name']
            attack_id = row['ATT&CK ID']

            # Add a string with this information to the kill chain list
            kill_chain.append(f"{phase_name}: {technique_name} ({attack_id})")

        # Convert the kill chain list to a string with newline characters between each item
        kill_chain_string = "\n".join(kill_chain)

        # Create System Message Template
        system_template = "You are a cybersecurity expert. Your task is to produce a comprehensive incident response testing scenario based on the information provided."
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        # Create Human Message Template
        human_template = ("""
                          **Background information:**
                          The company operates in the '{industry}' industry and is of size '{company_size}'. 
                          
                          **Threat actor information:**
                          Threat actor group '{selected_group_alias}' is planning to target the company using the following kill chain
                          {kill_chain_string}
                          
                          **Your task:**
                          Create an incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident 
                          response capabilities against the identified threat actor group. 
                          
                          Your response should be well structured and formatted using Markdown. Write in British English.
                          """)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Construct the ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # Format the prompt
        messages = chat_prompt.format_prompt(selected_group_alias=selected_group_alias, 
                                            kill_chain_string=kill_chain_string, 
                                            industry=industry, 
                                            company_size=company_size).to_messages()

# Error handling for group selection
except Exception as e:
    st.error("An error occurred: " + str(e))

try:
    if st.button('Generate Scenario'):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
        elif techniques_df.empty:
            st.info("Please select a threat group with associated Enterprise ATT&CK techniques.")
        else:
            response = generate_scenario(openai_api_key, messages)
            st.markdown("---")
            st.markdown(response.content)
            st.download_button(label="Download Scenario", data=response.content, file_name="incident_response_scenario.md", mime="text/markdown")
except Exception as e:
    st.error("An error occurred: " + str(e))