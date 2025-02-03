import os
import re
import streamlit as st
from app.process import (
    extract_docx,
    extract_pdf,
    extract_text,
    model_load,
    translator,
    translator_sec,
)

# **Page Configuration**
st.set_page_config(
    page_title="Translation Agent",
    page_icon="static/GIMO-logo.webp",  # Path to your logo
    layout="wide"
)

# **Custom Styling**
st.markdown(
    """
    <style>
        .title-container {
            text-align: center;
            margin-bottom: 10px;
        }
        .title-text {
            font-size: 28px;
            font-weight: bold;
            color: #4A90E2;
        }
        .stTextArea textarea {
            font-size: 16px !important;
            padding: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            width: 200px;
            display: block;
            margin: auto;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stImage img {
            max-width: 120px;
            display: block;
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title-container"><div class="title-text">GIMO Translation Agent</div></div>', unsafe_allow_html=True)

# **Helper Function: Translation Workflow**
def huanik(
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int,
    temperature: float,
    rpm: int,
    tone: int,
):
    if not source_text or not target_lang or not country:
        st.error("Please select all required options before translating.")
        return None

    try:
        model_load("OpenAI", "", "gpt-4o", "", temperature, rpm)  # Default values
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

    source_text = re.sub(r"(?m)^\s*$\n?", "", source_text)

    final_translation = translator(
        source_lang, target_lang, source_text, tone, country, max_tokens
    )

    return final_translation


# **Fix: File Upload Processing**
def read_doc(file):
    """Reads text from supported file formats."""
    file_type = file.name.split(".")[-1].lower()

    if file_type in ["pdf", "txt", "py", "docx", "json", "cpp", "md"]:
        if file_type == "pdf":
            return extract_pdf(file)
        elif file_type == "docx":
            return extract_docx(file)
        else:
            return file.read().decode("utf-8")  # Convert uploaded file to string
    else:
        st.error("Unsupported file type.")
        return ""


# **Sidebar Widgets for Configuration**
with st.sidebar:
    st.header("Translation Settings")

    # **Dropdowns for Language & Country**
    source_lang = st.selectbox("Source Language", ["English", "French", "Dutch", "German", "Romanian", "Italian", "Spanish", "Portuguese", "Finnish", "Danish", "Greek"], index=0)
    target_lang = st.selectbox("Target Language", ["", "English", "French", "Dutch", "German", "Romanian", "Italian", "Spanish", "Portuguese", "Finnish", "Danish", "Greek"], index=0)
    country = st.selectbox("Country", ["", "UK", "France", "Germany", "Switzerland", "Denmark", "Romania", "Italy", "Brazil", "Spain", "Greece", "Finalnd", "Netherlands"], index=0)
    
    # **Tone Selection with Dynamic Label**
    tone_labels = {
        1: "Very Informal",
        2: "Informal",
        3: "Neutral",
        4: "Formal",
        5: "Very Formal"
    }
    
    tone = st.slider("Translation Tone", min_value=1, max_value=5, value=3, step=1)
    st.markdown(f"**Selected Tone:** {tone} - {tone_labels[tone]}")

    # **Advanced Options (Collapsed by Default)**
    with st.expander("Advanced Options", expanded=False):
        max_tokens = st.slider("Max Tokens Per Chunk", min_value=512, max_value=2046, value=1000, step=8)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        rpm = st.slider("Requests Per Minute", min_value=1, max_value=1000, value=60, step=1)

# **Session State Initialization**
if "translation_output" not in st.session_state:
    st.session_state["translation_output"] = ""

# **Two-Column Layout (Left: Source, Right: Target)**
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Text")
    source_text = st.text_area("Enter text in English:", height=250, key="source_text")

# **File Upload (Now Below Left Panel)**
st.markdown("---")  # Adds a separator line
file = st.file_uploader("Upload File", type=["pdf", "txt", "py", "docx", "json", "cpp", "md"])
if file:
    source_text = read_doc(file)

# **Function to Perform Translation and Update Session State**
def translate():
    if not st.session_state.source_text.strip():
        st.warning("Please enter text to translate.")
        return

    if not target_lang or not country:
        st.warning("Please select a target language and country.")
        return

    with st.spinner("Translating... Please wait"):
        final_translation = huanik(
            source_lang, target_lang, st.session_state.source_text, country, max_tokens, temperature, rpm, tone
        )

        if final_translation:
            st.session_state.translation_output = final_translation  # Update session state


# **Translate Button with `on_click` to Modify State**
st.markdown("---")  # Adds a separator line
st.button("Translate", on_click=translate)

# **Displaying Translated Text**
with col2:
    st.subheader("Translated Text")
    st.text_area("Translated Output:", value=st.session_state.translation_output, height=250, key="translated_output")
