import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import chromadb.config

import openai
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma




# Initialize your chatbot components here
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# Streamlit app
def bubot():
    st.set_page_config(page_title="BuBot", page_icon=":robot_face:")

    # Add some CSS styles
    css = """
    <style>
    .chat-container {
        background-color: #f0f0f0;
        border-radius: 8px;
        padding: 20px;
    }
    .chat-bubble {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .bot-bubble {
        background-color: #e6f3ff;
        margin-left: auto;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.title("🤖 BuBot - Your AI Assistant")

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...")
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

    chat_history = []

    if "query" not in st.session_state:
        st.session_state.query = ""

    with st.container():
        query = st.text_input("Ask BuBot a question:", st.session_state.query, key="query")

        if query:
            result = chain({"question": query, "chat_history": chat_history})
            chat_history.append((query, result["answer"]))
            st.session_state.query = ""  # Clear the input field

    with st.container():
        st.subheader("Chat History")
        for user_query, bot_answer in chat_history:
            with st.container():
                st.markdown(f"<div class='chat-bubble'>You: {user_query}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble bot-bubble'>BuBot: {bot_answer}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    bubot()
