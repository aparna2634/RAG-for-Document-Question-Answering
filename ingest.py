from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def process_pdf(uploaded_file):
    """
    Processes an uploaded PDF file from Streamlit.
    Saves to a temp file, loads, and splits into chunks.
    """
    # 1. Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 2. Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # 3. Split into chunks
        # Requirements: chunk_size=500, chunk_overlap=100
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
