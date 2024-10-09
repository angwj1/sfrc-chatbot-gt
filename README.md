# SFRC Chatbot

## CHANGES MADE:
- Use GovTech API key
- Use GPT-4o (instead of GPT-4o-mini)
- Stick with text-embedding-3-small-prd-gcc2-lb (instead of upgrading to text-embedding-3-large due to better response -- weird)

## Problem Statement
Implementation of steel fibre reinforced concrete (SFRC) in underground infrastructure is new to most engineers. There are a series of new documents published recently, and engineers might not have the time/capacity to run through all these documents. However, it is important that they have a clear understanding of these documents because they have to review Contractor's proposal and safeguard agency's interests. 

 ## Proposed Solution
This Streamlit App is a Retrieval-Augmented Generation (RAG) Chatbot that serves to address any queries related to steel fibre reinforced concrete (SFRC) and provide the relevant document pages for further reading by user. The documents in the chatbot's knowledge bank include official Singapore Standard design code, material test code, and Agency's Particular Specifications. Several techniques were used to improve the Chatbot performance at different stages - pre-retrieval, retrieval, and post-retrieval. The performance of the Chatbot was evaluated using a synthetic Q&A dataset (specifically 69 Q&A pairs that are auto-generated with Ragas, but manually checked and verified by subject matter expert). Based on Ragas evaluation, the Chatbot achieved an answer correctness score of 71%, outperforming a standard chatbot by 7%. Potentially, this chatbot can be deployed to a public-facing one to address queries from consultants & contractors & industry players (though we have to cognisant that a chatbot can never be 100% accurate). The techniques used to improve the chatbot performance are listed below:
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
            
## Data
  - Singapore Standard SS674:2021
  - Design Guide for Fibre-Reinforced Concrete Structures to Singapore Standard SS674:2021
  - [Particular Specification] Steel Fibre Reinforced Concrete (SFRC) and Hybrid Segmental Lining
  - [Particular Specification] Design Requirements
  - [Particular Specification] Steel Fibre Reinforced Concrete (SFRC) for In-situ Casting
  - [EN 14721] Test method for metallic fibre concrete - Measuring the fibre content in fresh and hardened concrete
  - [EN 14651] Test method for metallic fibered concrete - Measuring the flexural tensile strength
  - [UNE 83515] Barcelona Test - Test method for determining the cracking strength, ductility and residual tensile strength of fibre reinforced concrete

## Value
  - Have a chatbot fully understand all the official documents
  - Consistent response and answer to queries about SFRC (based on official documents)
  - Customise the RAG chatbot to achieve better results (i.e. through more logical splitting/chunking of documents, other RAG techniques suitable for our technical documents)

