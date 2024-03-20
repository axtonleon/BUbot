import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import constants
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize your chatbot components here
os.environ["OPENAI_API_KEY"] = "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"

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #6B7280;
        padding: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        border: 2px solid #6B7280;
        padding: 10px 20px;
        background-color: #6B7280;
        color: white;
    }
    .stButton > button:hover {
        background-color: #808B96;
    }
</style>
""", unsafe_allow_html=True)

# Create a title for the app
st.title("Bubot - Your Personal Chatbot")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Create a text input for the user's query with a placeholder
query = st.text_input("Enter your query here", placeholder="Type your question here...")

# Create a button to send the query
if st.button("Ask Bubot"):
    if query:
        # Load the documents and create the index
        loader = DirectoryLoader(".", glob="*.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        # Assuming st.session_state.conversation is a list of tuples where each tuple is (sender, message)
        conversation_history = "\n".join([f"{sender}: {message}" for sender, message in st.session_state.conversation])

        # Create a prompt that includes the conversation history
        prompt = f"{conversation_history}\nUser: {query}"
        # Query the index and get the answer
        answer = index.query(prompt, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7))
        
        # Add the user's query and the bot's answer to the conversation history
        st.session_state.conversation.append(("User", query))
        st.session_state.conversation.append(("Bubot", answer))
        
        # Display the conversation history
        for sender, message in st.session_state.conversation:
            st.markdown(f"**{sender}:** {message}")
    else:
        st.warning("Please enter a query.")
