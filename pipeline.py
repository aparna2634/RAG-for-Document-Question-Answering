from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_llm():
    """
    Initializes and returns the Groq Llama 3 model.
    Loads API key from GROQ_API_KEY environment variable.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", # Updated for 2026 compatibility
        groq_api_key=api_key,
        temperature=0.1
    )

def format_docs(docs):
    """Formats retrieved documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(llm, vector_store):
    """
    Creates and returns a modern RAG chain using LCEL with Groq.
    Uses MMR (Maximal Marginal Relevance) for better chunk diversity.
    """
    from operator import itemgetter
    
    # Using MMR search to ensure we get a diverse set of chunks
    # (prevents getting 6 chunks of the same repetitive footer/header)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
    )
    
    # Professional RAG prompt for Llama 3
    template = """You are a professional assistant answering questions based on the provided document context.
Analyze the context thoroughly and provide a detailed, accurate answer.
If the information is not in the document, politely say so.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Core generation chain
    generation_chain = (
        {
            "context": itemgetter("query") | retriever | format_docs, 
            "question": itemgetter("query")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Parallel chain to return both result and sources
    rag_chain_with_sources = RunnableParallel(
        {
            "result": generation_chain, 
            "source_documents": itemgetter("query") | retriever
        }
    )
    
    return rag_chain_with_sources
