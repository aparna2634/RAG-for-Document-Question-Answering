from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
import os

# Configuration
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"

def get_embeddings():
    """
    Initializes and returns the HuggingFace embeddings model.
    """
    encode_kwargs = {'normalize_embeddings': True} # typical for bge
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs=encode_kwargs
    )
    return embeddings

def create_vector_store(chunks, embeddings, persist_dir=None):
    """
    Creates a persistent ChromaDB vector store from document chunks.
    Use a unique persist_dir to avoid file locks on Windows.
    """
    target_dir = persist_dir or PERSIST_DIRECTORY
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=target_dir
    )
    vector_db.persist()
    return vector_db

def get_vector_store(embeddings, persist_dir=None):
    """
    Loads an existing ChromaDB vector store.
    """
    target_dir = persist_dir or PERSIST_DIRECTORY
    if os.path.exists(target_dir):
        return Chroma(
            persist_directory=target_dir,
            embedding_function=embeddings
        )
    return None
