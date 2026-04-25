# RAG for Document Question Answering

A simple document question-answering app built with Python, LangChain, ChromaDB, HuggingFace Embeddings, and Gemini AI.

---

## Overview

Upload any PDF and ask questions about it. The system finds the most relevant parts of the document and uses Gemini to generate a grounded answer. If the answer is not in the document, it says so.

---

## Tech Stack

- **Python** — core language
- **LangChain** — pipeline orchestration
- **ChromaDB** — local vector store
- **HuggingFace Embeddings** — `BAAI/bge-small-en` for semantic search
- **Google Gemini 2.0 Flash** — answer generation via REST API
- **Streamlit** — web UI

---

## Features

- Upload a PDF and ask any question about it
- Semantic search retrieves the most relevant chunks
- Answers are strictly grounded in the document
- Returns "I don't know" when the answer is not found
- Clean, simple UI — no clutter

---

## Conclusion

This project demonstrates a complete RAG pipeline that answers questions from private documents without hallucination, using only free and open-source tools.
