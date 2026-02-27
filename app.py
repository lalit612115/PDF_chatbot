import os
from dotenv import load_dotenv
load_dotenv()
import warnings
import logging
import streamlit as st


from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma



warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')
if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])


@st.cache_resource
def get_vectorstore():
    pdf_name = "./pdf_analysis.pdf"
    loader = PyPDFLoader(pdf_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    documents = loader.load_and_split(text_splitter)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(
            model_name="all-MiniLM-L12-v2"
        )
    )

    return vectorstore

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything, you always give the best, 
           the most accurate and most precise answers. 
           Answer the following Question: 
           Context:{context}
           Question:{question}
           Start the answer directly. No small talk please""")

    model="llama-3.1-8b-instant"

    groq_chat = ChatGroq(

            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")
      
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = (
             {
                  "context": retriever,
                  "question": RunnablePassthrough()
                  }
            | groq_sys_prompt
            | groq_chat
            | StrOutputParser()
       )
        response = rag_chain.invoke(prompt)
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    except Exception as e:
        st.error(f"Error: {str(e)}")


