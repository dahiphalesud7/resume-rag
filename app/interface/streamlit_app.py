import streamlit as st
import os

from app.loader.pdf_loader import load_all_resumes
from app.processor.cleaner import clean_text
from app.embedder.text_embedder import get_embeddings, model
from app.vectorstore.faiss_handler import FAISSHandler
from app.vectorstore.bm25_handler import BM25Handler
from app.retriever.hybrid_retriever import create_hybrid_retriever

from langchain.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Resume Search RAG App", layout="wide")
st.title("📄 Resume Search RAG App")

# Use session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

folder_path = st.text_input("📁 Enter path to resumes folder:")

if folder_path and os.path.exists(folder_path):
    st.success("✅ Folder found.")
    
    if st.button("🚀 Process Resumes"):
        with st.spinner("Loading and processing..."):
            texts, metadatas = load_all_resumes(folder_path)
            texts = [clean_text(t) for t in texts]
            embeddings = get_embeddings(texts)

            faiss_handler = FAISSHandler(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            faiss_handler.create_index(texts, metadatas)

            bm25_handler = BM25Handler()
            bm25_handler.create_index(texts, metadatas)

            st.session_state.retriever = create_hybrid_retriever(
                faiss_handler.get_retriever(),
                bm25_handler.get_retriever()
            )
        st.success("✅ Resumes processed and index built.")

if st.session_state.retriever:
    query = st.text_input("🔍 Enter your query:")
    if query:
        with st.spinner("Searching..."):
            results = st.session_state.retriever.invoke(query)
            if results:
                st.markdown("### 🔎 Top Matching Resumes")
                for i, doc in enumerate(results, 1):
                    st.markdown(f"**Resume {i} ({doc.metadata['source']}):**\n\n{doc.page_content[:500]}...\n---")
            else:
                st.warning("No matching resumes found.")
