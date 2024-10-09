import streamlit as st
import pandas as pd
import json

from utility import check_password


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="> Methodology"
)

# ---------------------------------------------------------------------
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# ---------------------------------------------------------------------

##
# Methodology” Page:
# A comprehensive explanation of the data flows and implementation details.
# A flowchart illustrating the process flow for each of the use cases in the application. 

st.title("Methodology")
st.markdown("""
            Several techniques were used to improve the performance of the Chatbot (i.e. "Improved Chatbot") at different stages - pre-retrieval, retrieval, and post-retrieval. Based on Ragas evaluation, the Improved Chatbot achieved an answer correctness score of 71%, outperforming a standard chatbot by 7%. Two flowcharts, each detailing the techniques adopted in the Improved Chatbot and the standard chatbot, are presented below - the key differences between the chatbots are highlighted in red. The source code and the notebook can be found [here](https://github.com/angwj1/sfrc-chatbot).
            """)

with st.expander("Techniques used to improve the chatbot performance:: ", expanded=False):
    st.markdown("""
    - Pre-retrieval
        - Preprocess documents [to remove irrelevant pages & content]
        - Group pages together by documents, then perform semantic chunking [to identify logical breakpoints and create meaningful chunks]
        - Tag semantic chunks with page number using keyword search [to return the page number for each chunk]
    - Retrieval
        - Hybrid Search - combination of semantic search & keyword search [to capture unique keywords in technical documents]
    - Post-retrieval
        - Cross-encoder reranking of retrieved chunks with COHERE API [to identify the most relevant chunks]
        - Reorder retrieved chunks [to mitigate lost-in-the-middle effect]
        - Prompt Engineering 
            - temperature=0 [to return consistent result, grounded in context provided]
            - system messages [to prevent prompt injection] 
            - full chat history [to allow LLM to keep track of conversation with user]
            
        """)

st.write("Standard Chatbot Flowchart:")
st.image("gui_improved_gpt/image/standard_chatbot_flowchart.png")
st.write("  ")
st.write("Improved Chatbot Flowchart:")
st.image("gui_improved_gpt/image/improved_chatbot_flowchart.png")

