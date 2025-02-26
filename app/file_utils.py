import streamlit as st
import docx
import pymupdf

def extract_text(path):
    with open(path) as f:
        file_text = f.read()
    return file_text

def extract_pdf(path):
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_docx(path):
    doc = docx.Document(path)
    data = []
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)
    content = "\n\n".join(data)
    return content

def read_doc(file):
    """Reads text from supported file formats."""
    file_type = file.name.split(".")[-1].lower()

    if file_type in ["pdf", "txt", "py", "docx", "json", "cpp", "md"]:
        if file_type == "pdf":
            return extract_pdf(file)
        elif file_type == "docx":
            return extract_docx(file)
        else:
            return file.read().decode("utf-8")
    else:
        st.error("Unsupported file type.")
        return "" 