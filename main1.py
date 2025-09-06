import streamlit as st
import os

from app.loader.pdf_loader import load_all_resumes
from app.processor.cleaner import clean_text
from app.embedder.text_embedder import get_embeddings
from app.vectorstore.faiss_handler import FaissHandler
from app.vectorstore.bm25_handler import BM25Handler
from app.retriever.hybrid_retriever import HybridRetriever

# 🎯 Page setup
st.set_page_config(page_title="Resume Retriever", layout="wide")
st.title("🔍 Resume Retrieval System (BM25 + FAISS Hybrid)")

# 📂 Folder input
folder_path = st.text_input("📁 Enter folder path with PDF resumes:", "resumes")

# 🔄 Session state initialization
for key in ["texts", "metadata", "embeddings", "bm25", "faiss"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🚀 Process resumes
if st.button("🚀 Load and Process Resumes"):
    if not os.path.exists(folder_path):
        st.error("❌ Folder path does not exist.")
    else:
        # 1. Load resumes and extract text
        metadata, docs = load_all_resumes(folder_path)

        # 2. Clean the text content
        cleaned_texts = [clean_text(doc) for doc in docs]

        # Store in session
        st.session_state.texts = cleaned_texts
        st.session_state.metadata = metadata

        # 3. BM25 Handler
        st.session_state.bm25 = BM25Handler(cleaned_texts, metadata)

        # 4. Generate embeddings
        embeddings = get_embeddings(cleaned_texts)
        st.session_state.embeddings = embeddings

        # 5. FAISS Handler
        faiss = FaissHandler(cleaned_texts, metadata)
        st.session_state.faiss = faiss

        st.success("✅ Resumes processed and retrievers are ready!")

# 🔎 Search interface
st.markdown("---")
query = st.text_input("💼 Enter job description or query:")

if query and st.button("🔍 Retrieve Matching Resumes"):
    if st.session_state.texts is None or st.session_state.embeddings is None:
       st.warning("⚠️ Please process resumes before searching.")

    else:
        hybrid = HybridRetriever(
            bm25_handler=st.session_state.bm25,
            faiss_handler=st.session_state.faiss
        )

        results = hybrid.retrieve(query)

    if results:
       st.markdown("### 📄 Top Matching Resumes")

    for i, res in enumerate(results, 1):
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; margin-bottom: 20px; border-radius: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
            <h4>📄 Resume {i}: <code>{res['filename']}</code></h4>
            <div style="font-size: 15px; line-height: 1.6; white-space: pre-wrap;">{res['content']}</div>
        </div>
        """, unsafe_allow_html=True)



