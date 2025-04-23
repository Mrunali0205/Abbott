import streamlit as st
import os
from main import AzureAIClient, UserQueryProcessor, EmbeddingComparer, azure_client, few_shot_examples
import json

# Set up the Streamlit page
st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")

# Title of the app
#st.title("Standard Occupational Classification")
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <style>
    .title-css{
        font-size: 3em;
        font-weight: bold;
        color: "white";
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 20px;
    }
    .chat-message {
        border-radius: 12px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
    }
    .user-message {
        align-self: flex-end;
        background-color: #dcf8c6;
        color: #333;
    }
    .bot-message {
        align-self: flex-start;
        background-color: #f1f0f0;
        color: #333;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #fff;
        padding: 10px;
    }
    </style>
    <h1 class="title-css">Standard Occupational Classification</h1>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    if message["sender"] == "user":
        st.markdown(f'<div class="chat-container"><div class="chat-message user-message">{message["text"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container"><div class="chat-message bot-message">{message["text"]}</div></div>', unsafe_allow_html=True)


# Create a text input for user message

#user_query = st.text_area("User-Input", key="user_input", placeholder="Enter your information here", height=150)
user_query = st.chat_input("Enter your information here")


# If the user submits a message
if user_query:
        st.session_state.messages.append({"sender": "user", "text": user_query})
        st.markdown(f'<div class="chat-container"><div class="chat-message user-message">{user_query}</div></div>', unsafe_allow_html=True)
        with st.spinner("Processing your request... Please wait."):
            query_processor = UserQueryProcessor(azure_client, few_shot_examples)
            
           # user_query = 'Prepare operational budgets for green energy or other green operations, Integrity -  Job requires being honest and ethical,Portable data input terminals -  Dataloggers, Monitor geothermal operations, using programmable logic controllers.'

            # Process query and generate embeddings
            classified_query = query_processor.process_query(user_query)
            user_embeddings = query_processor.create_embeddings(classified_query)

            # Compare embeddings
            comparer = EmbeddingComparer()
            merged_embed_dir = 'MergedEmbeddings'
            closest_file = comparer.compare_embeddings(user_embeddings, merged_embed_dir)

            json_file_path = os.path.join('JSON', f"{closest_file}.json")

            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                    # Extract relevant information
                    soc_code = data.get('SocCode', 'N/A')
                    occupation = data.get('Occupation', 'N/A')
                    reported_job_titles = data.get('ReportedJobTitles', 'N/A')

                    if isinstance(reported_job_titles, str) and reported_job_titles.lower().startswith("sample of reported job titles:"):
                        reported_job_titles = reported_job_titles[len("sample of reported job titles:"):].strip()
                    
                # Display the information in a formatted way inside st.success
                    response_text = (
                        f"\n\n"
                        f"**SOC Code :** {soc_code}\n\n"
                        f"**Occupation :** {occupation}\n\n"
                        f"**Reported Job Titles :** {reported_job_titles}"
                    )
                st.session_state.messages.append({"sender": "bot", "text": response_text})
                st.markdown(f'<div class="chat-container"><div class="chat-message bot-message">{response_text}</div></div>', unsafe_allow_html=True)

            else:
                st.error(f"JSON file for SOC code {soc_code} not found.")


            
            
    