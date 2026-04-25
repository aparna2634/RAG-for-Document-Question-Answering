import streamlit as st
import os
from dotenv import load_dotenv
from ingest import process_pdf
from retriever import get_embeddings, create_vector_store, get_vector_store
from pipeline import get_llm, get_rag_chain

# Load environment variables
load_dotenv()

# Streamlit Page Configuration
st.set_page_config(page_title="Professional RAG System (Groq)", layout="centered")

# App Header
st.title("RAG for Question Answering")
st.markdown("Upload a PDF and ask questions. Powered by **Groq (Llama 3)** for ultra-fast, high-quality answers.")

# --- API Key Management (Hidden) ---
# Supports both GOOGLE_API_KEY (legacy) and GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Resource Caching ---
@st.cache_resource
def load_embeddings():
    return get_embeddings()

embeddings = load_embeddings()

# --- Main Document Upload Section ---
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

vector_db = None
if uploaded_file is not None:
    # IMPORTANT: Use a unique directory based on filename to avoid Windows lock issues
    # and ensure document isolation.
    import hashlib
    file_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
    session_db_path = os.path.join("./chroma_db", file_id)
    
    with st.status("Reading document...", expanded=False) as status:
        chunks = process_pdf(uploaded_file)
        vector_db = create_vector_store(chunks, embeddings, persist_dir=session_db_path)
        status.update(label="Document Processed!", state="complete")
    st.success(f"Focused on: {uploaded_file.name}")
else:
    # Fallback to default or stay None
    vector_db = None

# --- Main QA Interface ---
if not groq_api_key:
    st.error("❌ Groq API Key not found. Please add `GROQ_API_KEY` to your `.env` file.")
    st.info("Get a free key from https://console.groq.com/keys")
elif vector_db is None:
    st.info("💡 Please upload a PDF to start.")
else:
    st.divider()
    user_query = st.text_input("What would you like to know about the document?", placeholder="e.g. Summarize the main points.")
    
    if user_query:
        with st.spinner("Groq is analyzing the document..."):
            try:
                # Get LLM and Chain
                llm = get_llm()
                if llm is None:
                    st.error("Could not initialize Groq LLM. check your API key.")
                else:
                    rag_chain = get_rag_chain(llm, vector_db)
                    
                    # Get answer
                    response = rag_chain.invoke({"query": user_query})
                    
                    # Display Answer
                    st.subheader("Answer")
                    st.success(response["result"])
            except Exception as e:
                st.error(f"Error calling Groq API: {e}")
                st.info("Check if your API key is correct and valid in the `.env` file.")
