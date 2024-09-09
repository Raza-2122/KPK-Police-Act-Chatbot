import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import textwrap

# Load the API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Ensure you set this in your environment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize embeddings and language model
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Load PDF document
pdf_loader = PyPDFLoader(r"THE_KHYBER_PAKHTUNKHWA_POLICE_ACT_2017.pdf")
docs = pdf_loader.load()

# Convert docs into vectorstore
vectorstore = Chroma.from_documents(docs, gemini_embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Function to wrap text
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Chatbot template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<h1 class="centered">🤖 KPK Police Act 2017 Chatbot </h1>', unsafe_allow_html=True)
st.markdown('<p class="centered">Welcome to the KPK Police Act 2017 Chatbot! Ask me anything related to the Act📜</p>', unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .chatbox-container {
        display: flex;
        align-items: center;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 5px;
        margin-top: 10px;
    }
    .chatbox-input {
        flex: 1;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    .chatbox-send {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    .chatbox-send:hover {
        background-color: #45a049;
    }
    .response {
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        background-color: #f1f1f1;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit input field and button
user_question = st.text_input("💬 Type your question here...", key="question")
if st.button("Send 📨"):
    if user_question:
        st.write("You asked: ", user_question)
        
        # Process the query using LangChain
        response = chain.invoke(user_question)
        
        # Display the response in bold
        st.write("Response:")
        st.markdown(f"**{wrap_text(response)}**", unsafe_allow_html=True)

# Developer credits
st.write("---")
st.write("<b>Developers: Yasir Huassain | Syed Qasim Raza Fatimi</b>", unsafe_allow_html=True)
