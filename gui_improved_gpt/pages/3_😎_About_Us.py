import streamlit as st
import os
import json

from utility import check_password

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="> About Us"
)

# ---------------------------------------------------------------------
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()
# ---------------------------------------------------------------------

# load resources_full.json into a python dict (mapping each filename to its document title)
folder_path = 'gui/st_data'
filename = 'resources_full.json'
file_path = os.path.join(folder_path, filename)
with open(file_path) as f:
    resource_dictionary  = json.load(f)    
    
    
# convert filename to its document title
def filename2title(filename):
    for resource in resource_dictionary:
        if resource["filename"] == filename:
            return (resource["title"])

# endregion <--------- Streamlit App Configuration --------->
# About Us” Page:
# A detailed page outlining the project scope, objectives, data sources, and features.
st.title("About this App")
st.markdown("""
            This Streamlit App is a Retrieval-Augmented Generation (RAG) Chatbot that serves to address any queries related to steel fibre reinforced concrete (SFRC) and provide the relevant document pages for further reading by user. The documents in the chatbot's knowledge bank include official Singapore Standard design code, material test code, and LTA Particular Specifications. Several techniques were used to improve the Chatbot performance at different stages - pre-retrieval, retrieval, and post-retrieval. The performance of the Chatbot was evaluated using a synthetic Q&A dataset (specifically 69 Q&A pairs that are auto-generated with Ragas, but manually checked and verified by subject matter expert). Based on Ragas evaluation, the Chatbot achieved an answer correctness score of 71%, outperforming a standard chatbot by 7%. For a detailed comparison of the techniques adopted in the improved Chatbot and the standard chatbot, please refer to the "Methodology" page. The source code and the notebook can be found [here](https://github.com/angwj1/sfrc-chatbot).         
            """)

with st.expander("Documents in the chatbot's knowledge bank: ", expanded=False):
    for resource in resource_dictionary:
        st.markdown(f"""- {resource["title"]}""")