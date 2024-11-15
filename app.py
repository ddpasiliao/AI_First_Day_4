import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
from PIL import Image

import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention


warnings.filterwarnings('ignore')

st.set_page_config(page_title="MaxLoad: Load Optimization Assistant", page_icon="🏴‍☠️", layout="wide")


# Load the logistics-related image
try:
    maxload_path = "./MaxLoad_bg.jpg"
    maxload_image = Image.open(maxload_path)
except FileNotFoundError:
    st.error("MaxLoad image not found. Please ensure the file 'MaxLoad_bg.jpg' is in the same directory as this script.")

with st.sidebar:
    openai.api_key = st.text_input("OpenAI API Key", type="password")
    if not (openai.api_key.startswith('sk') and len(openai.api_key) == 164):
        st.warning("Please enter a valid OpenAI API key", icon="⚠️")
    else:
        st.success("API key is valid", icon="✅")

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()


# System prompt for better accuracy
system_prompt = """
You are MaxLoad, an expert logistics loading assistant dedicated to maximizing load efficiency and ensuring safe, accurate loading procedures for transportation. Your primary goal is to provide precise, well-organized, and practical guidance for load planning and logistics optimization. Follow these guidelines to excel in assisting with loading tasks:

Load-Based Responses:

    Source Information: Base all recommendations on logistics best practices, load capacity guidelines, and safety regulations.
    No Speculation: Avoid assumptions about load requirements. Focus on the provided parameters such as weight limits, dimensions, and transportation specifics.

Clarity and Detail:

    Concise yet Comprehensive: Be concise but thorough, ensuring each response contains necessary details to maximize loading efficiency and safety.
    Organized Guidance: Present instructions in a logical sequence, making them clear and actionable for the user.

Handling Uncertainty:

    Ambiguity: If a question lacks specifics or requires clarification, respond with: “Please provide additional details to ensure accurate guidance.”
    Unanswerable Scenarios: For requests beyond typical logistics parameters, acknowledge limitations, suggesting alternative actions where possible.

User Instructions:

    Encouragement to Ask: Encourage users to specify load details like cargo type, weight, dimensions, or vehicle type for optimal advice.
    Clarification Requests: If the user’s question is vague, ask specific follow-up questions to ensure complete and useful answers.

Review and Accuracy:

    Fact-Checking: Verify all advice aligns with logistics standards, load regulations, and relevant safety measures before responding.
"""

# Function to get the answer from OpenAI
def get_maxload_answer(query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    try:
        # Making the OpenAI call with a lower temperature for accuracy
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2  # Lower temperature for more focused and accurate answers
        )
        answer = response.choices[0].message.content.strip()
        
        # Validate answer length or content to retry if off-track
        if "load" not in answer and len(answer.split()) < 10:
            return "I'm unable to answer that question accurately."
        return answer

    except Exception as e:
        return f"Error: {e}"

# Main content container with styling
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.title("MaxLoad: Load Optimization Assistant")
st.write("Efficient Loading Solutions for Maximum Capacity and Safety")

# Input for user questions related to load optimization
load_input = st.text_input("Ask about load optimization or logistics", placeholder="Enter your question here")
submit_button = st.button("Get Loading Advice")

if submit_button and load_input:
    with st.spinner("Calculating optimal load..."):
        # Get the answer using MaxLoad's API or logic
        response = get_load_advice(load_input)
        st.write(response)
st.markdown('</div>', unsafe_allow_html=True)

