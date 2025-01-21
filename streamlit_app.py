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
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stImage img {
            max-width: 120px;  /* Adjusts logo size */
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
    endpoint: str,
    base: str,
    model: str,
    api_key: str,
    choice: bool,
    endpoint2: str,
    base2: str,
    model2: str,
    api_key2: str,
    source_lang: str,
    target_lang: str,
    source_text: str,
    country: str,
    max_tokens: int,
    temperature: float,
    rpm: int,
):
    if not source_text or source_lang == target_lang:
        st.error("Please check that the content or options are entered correctly.")
        return None

    try:
        model_load(endpoint, base, model, api_key, temperature, rpm)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

    source_text = re.sub(r"(?m)^\s*$\n?", "", source_text)

    if choice:
        _, _, final_translation = translator_sec(
            endpoint2=endpoint2,
            base2=base2,
            model2=model2,
            api_key2=api_key2,
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            country=country,
            max_tokens=max_tokens,
        )
    else:
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
            # Convert uploaded file to string (for text-based files)
            return file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        return ""


# **Sidebar Widgets for Configuration**
with st.sidebar:
    st.header("Settings")
    endpoint = st.selectbox(
        "Translation Endpoint", ["OpenAI", "Groq", "TogetherAI", "Ollama", "CUSTOM"], index=0
    )
    model = st.text_input("Model", value="gpt-4o")
    api_key = st.text_input("API Key", type="password")
    base = st.text_input("Base URL (if CUSTOM)", value="") if endpoint == "CUSTOM" else ""

    choice = st.checkbox("Use Additional Endpoint")
    if choice:
        st.subheader("Additional Endpoint Settings")
        endpoint2 = st.selectbox(
            "Additional Endpoint", ["OpenAI", "Groq", "TogetherAI", "Ollama", "CUSTOM"], index=0
        )
        model2 = st.text_input("Model (2nd Endpoint)", value="gpt-4o")
        api_key2 = st.text_input("API Key (2nd Endpoint)", type="password")
        base2 = st.text_input("Base URL (2nd Endpoint)", value="") if endpoint2 == "CUSTOM" else ""

    source_lang = st.text_input("Source Language", value="English")
    target_lang = st.text_input("Target Language", value="Spanish")
    country = st.text_input("Country", value="Argentina")

    st.subheader("Advanced Options")
    max_tokens = st.slider("Max Tokens Per Chunk", min_value=512, max_value=2046, value=1000, step=8)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    rpm = st.slider("Requests Per Minute", min_value=1, max_value=1000, value=60, step=1)


# **Two-Column Layout (Left: Source, Right: Target)**
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Text")
    file = st.file_uploader("Upload File", type=["pdf", "txt", "py", "docx", "json", "cpp", "md"])
    if file:
        source_text = read_doc(file)
    else:
        source_text = st.text_area("Enter text in English:", height=250)

with col2:
    st.subheader("Translated Text")
    target_text_placeholder = st.empty()  # Placeholder for translation output

# **Translate Button (Centered)**
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
translate_button = st.button("Translate")
st.markdown("</div>", unsafe_allow_html=True)

# **Translation Execution**
if translate_button:
    if not source_text.strip():
        st.warning("Please enter text to translate.")
    else:
        final_translation = huanik(
            endpoint, base, model, api_key,
            choice, endpoint2 if choice else "", base2 if choice else "", model2 if choice else "", api_key2 if choice else "",
            source_lang, target_lang, source_text, country, max_tokens, temperature, rpm
        )

        if final_translation:
            target_text_placeholder.text_area("Translated Output:", value=final_translation, height=250)
