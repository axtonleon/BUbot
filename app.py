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
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

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

# Create a title for the app
st.title("Bubot - Your Personal Chatbot")

# Create a text input for the user's query with a placeholder
query = st.text_input("Enter your query here", placeholder="Type your question here...")

# Create a button to send the query
if st.button("Ask Bubot"):
    if query:
        # Load the documents and create the index
        loader = DirectoryLoader(".", glob="*.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        
        # Query the index and get the answer
        answer = index.query(query, llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7))
        
        # Display the answer with a nice styling
        st.markdown(f"## Answer:")
        st.markdown(f"{answer}")
    else:
        st.warning("Please enter a query.")
