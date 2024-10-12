from dotenv import load_dotenv 
if not load_dotenv(override=True): # only apply the following code to use pysqlite3 if not running in local machine (i.e. run on streamlit community cloud)
    __import__('pysqlite3')
    import sys,os
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    # https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/4

import streamlit as st
# import os
# from dotenv import load_dotenv 
# from openai import OpenAI
# import time

from utility import check_password
from utility import generate_llm_response_simple, generate_llm_response_from_conversation
from utility import filename2title

from langchain.retrievers import EnsembleRetriever
import cohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_chroma import Chroma
import pickle
import os
import json
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Ask SFRC Chatbot"
)

# ---------------------------------------------------------------------
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
    
# ---------------------------------------------------------------------

# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
# https://getemoji.com/
# https://blog.streamlit.io/introducing-multipage-apps/#tips-and-tricks


####### SET-UP #######
# # set up openai api key and cohere api key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY_GOVTECH']
os.environ['COHERE_API_KEY'] = st.secrets['COHERE_API_KEY']
# embedding model that we will use for the session
embeddings_model = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://litellm.govtext.gov.sg/",
    default_headers={"user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/81.0"},        
    model='text-embedding-3-large-prd-gcc2-lb', 
    # model='text-embedding-3-small-prd-gcc2-lb',
    )

# llm to be used in RAG pipeplines in this notebook
llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="https://litellm.govtext.gov.sg/",
    default_headers={"user-agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/81.0"},
    model = "gpt-4o-prd-gcc2-lb", 
    # model = "gpt-4o-mini-prd-gcc2-lb",
    temperature=0,
    seed=42,
    )

# set up openai api key and cohere api key
# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# os.environ['COHERE_API_KEY'] = st.secrets['COHERE_API_KEY']
# # embedding model that we will use for the session
# embeddings_model = OpenAIEmbeddings(model='text-embedding-3-large')
# # llm to be used in RAG pipeplines in this notebook
# llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# COHERE client to be used for cross-encoder reranking
co = cohere.Client()

# load vector database
if not load_dotenv(override=True): # only apply the following code to use pysqlite3 if not running in local machine (i.e. run on streamlit community cloud)
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    # https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/4
vectordb = Chroma(
    embedding_function=embeddings_model,
    collection_name="semantic_splitter_improved_embeddings", # one database can have multiple collections
    persist_directory="gui_improved_gpt/st_output/vector_db"
)
# Set up vector search 
vectorstore_retriever = vectordb.as_retriever(search_kwargs={"k": 20})

# load bm25 object
folder_path = 'gui_improved_gpt/st_output/bm25/'
file_name = 'semantic_bm25_improved_embeddings'
file_path = os.path.join(folder_path, file_name)
with open(file_path, 'rb') as bm25result_file:
    keyword_retriever = pickle.load(bm25result_file)
# Set up keyword search
keyword_retriever.k = 20

# Rerank the results of the constituent retrievers based on the Reciprocal Rank Fusion (RRF) algorithm. Only take the top 10 documents and rerank again using COHERE cross encoder.
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retriever,
                                                    keyword_retriever],
                                        weights=[0.5, 0.5])

# load resources_full.json into a python dict (mapping each filename to its document title)
folder_path = 'gui_improved_gpt/st_data'
filename = 'resources_full.json'
file_path = os.path.join(folder_path, filename)
with open(file_path) as f:
    resource_dictionary  = json.load(f)

##################################

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "full_messages" not in st.session_state:
    st.session_state.full_messages = []
    
c1,c2= st.columns([7,2])
c1.title("🤖 Ask SFRC chatbot")
if c2.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.full_messages = []

with st.expander("Disclaimer: ", expanded=False):
    st.write("""
IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.
""")
    
# reproduce the full messages (including metadata and context) when script reruns in the same session
for message in st.session_state.full_messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"]):
            st.empty()   # this line of code removes the ghost text
            response, reranked_docs = message["content"]
            metadatas = [doc.metadata for doc in reranked_docs] 
            contexts = [doc.page_content for doc in reranked_docs]
            
            st.markdown(response)
            with st.expander("Source of information: ", expanded=False):
                for metadata in metadatas: 
                    st.write(f"document: {filename2title(resource_dictionary, metadata['source'])} ~~ on page {metadata['page']}")
            # with st.expander("Full details of information: ", expanded=False):
            #     for context in contexts: 
            #         st.write(context)
        
    else:
        with st.chat_message(message["role"]):
            st.empty()   # this line of code removes the ghost text
            st.markdown(message["content"])

# receive query input from user
prompt = st.chat_input("Please enter your query.")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Generating response...'):
            full_convo_history = st.session_state.messages
            current_query = prompt       
            reranked_docs, full_context, response = generate_llm_response_from_conversation(llm, ensemble_retriever, co, full_convo_history, current_query)
            # reranked_docs, full_context, response = generate_llm_response_simple(current_query)
            
            # exclude information source if user query is not unrelated
            if "I am sorry, but I cannot assist with that" in response:
                reranked_docs = []
            metadatas = [doc.metadata for doc in reranked_docs] 
            contexts = [doc.page_content for doc in reranked_docs]
            
            st.markdown(response)

            with st.expander("Source of information: ", expanded=False):
                for metadata in metadatas: 
                    st.write(f"document: {filename2title(resource_dictionary, metadata['source'])} ~~ on page {metadata['page']}")
            # with st.expander("Full details of information: ", expanded=False):
            #     for context in contexts: 
            #         st.write(context)
    
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.full_messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.full_messages.append({"role": "assistant", "content": [response, reranked_docs]})
            