import os

__import__('pysqlite3')
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import chromadb


import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma




# Initialize your chatbot components here
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

# Streamlit app title
st.title('Bubot: Your Personal Chatbot')

# Streamlit sidebar for API key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', value=os.environ.get("OPENAI_API_KEY", ""))

# Streamlit form for user input
with st.form(key='bubot_form'):
    query = st.text_input('Enter your question:')
    submit_button = st.form_submit_button(label='Ask Bubot')

# Initialize the chatbot
if submit_button:
    PERSIST = False # Assuming you want to disable persistence for now
    chat_history = []

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    result = chain({"question": query, "chat_history": chat_history})
    st.write(result['answer'])

    chat_history.append((query, result['answer']))
