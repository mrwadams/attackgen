import os

import pandas as pd
import streamlit as st
from langchain.callbacks.manager import collect_runs
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langsmith import Client
from mitreattack.stix20 import MitreAttackData

# Add environment variables for LangSmith
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "AttackGen"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Initialize the LangSmith client
client = Client()

# Check if 'openai_api_key' exists in the session state
if "openai_api_key" not in st.session_state:
    st.error("""
             OpenAI API key not found!
             
             Please go to the `Welcome` page and enter your OpenAI API key to continue.
             """)
    st.stop()
else:
    openai_api_key = st.session_state["openai_api_key"]

if "scenario_generated" not in st.session_state:
    st.session_state["scenario_generated"] = False
    
industry = st.session_state["industry"]
company_size = st.session_state["company_size"]

st.set_page_config(
    page_title="Generate Scenario",
    page_icon="🎭",
)

def generate_scenario(openai_api_key, messages):
    try:
        with st.status('Generating scenario...', expanded=True):
            st.write("Initialising AI model.")
            llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=False)
            st.write("Model initialised. Generating scenario, please wait.")

            with collect_runs() as cb:
                response = llm.generate(messages=[messages])
                run_id1 = cb.traced_runs[0].id # Get run_id from the callback

            st.write("Scenario generated successfully.")
            st.session_state['run_id'] = run_id1 # Store the run ID in the session state
        return response
    except Exception as e:
        st.error("An error occurred while generating the scenario: " + str(e))
        return None

# Load and cache the MITRE ATT&CK data
@st.cache_resource
def load_attack_data():
    attack_data = MitreAttackData("./data/enterprise-attack.json")
    return attack_data

attack_data = load_attack_data()

# Load and cache the list of threat actor groups
@st.cache_resource
def load_groups():
    groups = pd.read_json("./data/groups.json")
    return groups

groups = load_groups()


st.markdown("# <span style='color: #1DB954;'>Generate Threat Group Scenario🛡️</span>", unsafe_allow_html=True)

st.markdown("""
            ### Select a Threat Actor Group

            Use the drop-down selector below to select a threat actor group from the MITRE ATT&CK framework. 
            
            You can then optionally view all of the Enterprise ATT&CK techniques associated with the group and/or the group's page on the MITRE ATT&CK site.
            """)

selected_group_alias = st.selectbox("Select a threat actor group for the scenario",
                                     sorted(groups['group'].unique()),placeholder="Select Group", index=17, label_visibility="hidden") # Set APT41 as the default group as the default group has no Enterprise ATT&CK techniques

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
Create an incident response testing scenario based on the information provided. The goal of the scenario is to test the company's incident response capabilities against the identified threat actor group. 

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
st.markdown("")
st.markdown("""
            ### Generate a Scenario

            Click the button below to generate a scenario based on the selected threat actor group. A selection of the group's known techniques will be chosen at random and used to generate the scenario.

            It normally takes between 30-50 seconds to generate a scenario. ⏱️
            """)

try:
    if st.button('Generate Scenario'):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
        elif not industry:
            st.info("Please select your company's industry to continue.")
        elif not company_size:
            st.info("Please select your company's size to continue.")
        elif techniques_df.empty:
            st.info("Please select a threat group with associated Enterprise ATT&CK techniques.")
        else:
            response = generate_scenario(openai_api_key, messages)
            st.markdown("---")
            if response is not None:
                st.session_state['scenario_generated'] = True
                scenario_text = response.generations[0][0].text
                st.session_state['scenario_text'] = scenario_text  # Store the generated scenario in the session state
                st.markdown(scenario_text)
                st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    else:
            # If a scenario has been generated previously, display it
            if 'scenario_text' in st.session_state and st.session_state['scenario_generated']:
                st.markdown("---")
                st.markdown(st.session_state['scenario_text'])
                st.download_button(label="Download Scenario", data=st.session_state['scenario_text'], file_name="threat_group_scenario.md", mime="text/markdown")

    # Create a placeholder for the feedback message
    feedback_placeholder = st.empty()

    # Show the thumbs_up and thumbs_down buttons only when a scenario has been generated
    st.markdown("---")
    if st.session_state.get('scenario_generated', True):
        st.markdown("Rate the scenario to help improve this tool.")
        col1, col2, col3 = st.columns([0.5,0.5,5])
        with col1:
            thumbs_up = st.button("👍")
            if thumbs_up:
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_type_str = "positive"
                        score = 1  # or 0
                        comment = ""

                        # Record the feedback
                        feedback_record = client.create_feedback(
                            run_id,
                            feedback_type_str,
                            score=score,
                            comment=comment,
                        )
                        st.session_state.feedback = {
                            "feedback_id": str(feedback_record.id),
                            "score": score,
                        }
                        # Update the feedback message in the placeholder
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        # Update the feedback message in the placeholder
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    # Update the feedback message in the placeholder
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")

        with col2:
            thumbs_down = st.button("👎")
            if thumbs_down:
                try:
                    run_id = st.session_state.get('run_id')
                    if run_id:
                        feedback_type_str = "negative"
                        score = 0  # or 0
                        comment = ""

                        # Record the feedback
                        feedback_record = client.create_feedback(
                            run_id,
                            feedback_type_str,
                            score=score,
                            comment=comment,
                        )
                        st.session_state.feedback = {
                            "feedback_id": str(feedback_record.id),
                            "score": score,
                        }
                        # Update the feedback message in the placeholder
                        feedback_placeholder.success("Feedback submitted. Thank you.")
                    else:
                        # Update the feedback message in the placeholder
                        feedback_placeholder.warning("No run ID found. Please generate a scenario first.")
                except Exception as e:
                    # Update the feedback message in the placeholder
                    feedback_placeholder.error(f"An error occurred while creating feedback: {str(e)}")

except Exception as e:
    st.error("An error occurred: " + str(e))

# Add a back button
link_to_homepage = "/"

st.markdown(
    f'<a href="{link_to_homepage}" style="display: inline-block; padding: 5px 20px; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">⬅️ Back</a>',
    unsafe_allow_html=True
)