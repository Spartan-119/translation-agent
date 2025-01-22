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
st.set_page_config(page_title="Translation Agent", layout="wide")

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

# **Logo (Smaller & Centered)**
logo_path = "static/GIMO-logo.webp"
if os.path.exists(logo_path):
    st.image(logo_path, use_container_width=False)

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

    _, _, final_translation = translator(
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
        max_tokens=max_tokens,
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
    source_lang = st.selectbox("Source Language", ["English", "French", "Dutch", "German", "Romanian", "Italian", "Spanish", "Portuguese"], index=0)
    target_lang = st.selectbox("Target Language", ["", "English", "French", "Dutch", "German", "Romanian", "Italian", "Spanish", "Portuguese"], index=0)
    country = st.selectbox("Country", ["", "UK", "France", "Germany", "Switzerland", "Denmark", "Romania", "Italy", "Brazil", "Spain"], index=0)

    st.subheader("Advanced Options")
    max_tokens = st.slider("Max Tokens Per Chunk", min_value=512, max_value=2046, value=1000, step=8)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    rpm = st.slider("Requests Per Minute", min_value=1, max_value=1000, value=60, step=1)

# **Two-Column Layout (Left: Source, Right: Target)**
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Text")
    source_text = st.text_area("Enter text in English:", height=250)

with col2:
    st.subheader("Translated Text")
    target_text_placeholder = st.empty()  # Placeholder for translation output

# **File Upload (Now Below Left Panel)**
st.markdown("---")  # Adds a separator line
file = st.file_uploader("Upload File", type=["pdf", "txt", "py", "docx", "json", "cpp", "md"])
if file:
    source_text = read_doc(file)

# **Translate Button (Now Below Right Panel)**
st.markdown("---")  # Adds a separator line
translate_button = st.button("Translate")

# **Translation Execution**
if translate_button:
    if not source_text.strip():
        st.warning("Please enter text to translate.")
    elif not target_lang or not country:
        st.warning("Please select a target language and country.")
    else:
        final_translation = huanik(
            source_lang, target_lang, source_text, country, max_tokens, temperature, rpm
        )

        if final_translation:
            target_text_placeholder.text_area("Translated Output:", value=final_translation, height=250)
