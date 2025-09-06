import streamlit as st
import os

from app.loader.pdf_loader import load_all_resumes
from app.processor.cleaner import clean_text
from app.embedder.text_embedder import get_embeddings
from app.vectorstore.faiss_handler import FaissHandler
from app.vectorstore.bm25_handler import BM25Handler
from app.retriever.hybrid_retriever import HybridRetriever
from app.llm.perplexity_llm import query_perplexity_llm
from app.ats_scorer.ats_scorer import compute_ats_score


# 🎯 Page setup
st.set_page_config(page_title="Resume Retriever", layout="wide")
st.title(" Resume Retrieval System (BM25 + FAISS Hybrid + Perplexity LLM)")

# 📂 Folder input
folder_path = st.text_input("Enter folder path with PDF resumes:", "resumes")

# 🔄 Session state initialization
for key in ["texts", "metadata", "embeddings", "bm25", "faiss"]:
    if key not in st.session_state:
        st.session_state[key] = None

# 🚀 Process resumes
if st.button(" Load and Process Resumes"):
    if not os.path.exists(folder_path):
        st.error(" Folder path does not exist.")
    else:
        metadata, docs = load_all_resumes(folder_path)
        cleaned_texts = [clean_text(doc) for doc in docs]

        st.session_state.texts = cleaned_texts
        st.session_state.metadata = metadata

        st.session_state.bm25 = BM25Handler(cleaned_texts, metadata)
        st.session_state.embeddings = get_embeddings(cleaned_texts)
        st.session_state.faiss = FaissHandler(cleaned_texts, metadata)

        st.success("Resumes processed and retrievers are ready!")

# 🔎 Search interface
st.markdown("---")
query = st.text_input(" Enter job description or query:")

if query and st.button(" Retrieve Matching Resumes"):
    if st.session_state.texts is None or st.session_state.embeddings is None:
        st.warning("Please process resumes before searching.")
    else:
        hybrid = HybridRetriever(
            bm25_handler=st.session_state.bm25,
            faiss_handler=st.session_state.faiss
        )
        results = hybrid.retrieve(query)
        for res in results:
            res["ats_score"] = compute_ats_score(res["content"], query)
        results.sort(key=lambda x: x['ats_score'], reverse=True)


              # ✅ Store in session_state
        st.session_state.results = results
        st.session_state.resume_texts = [r['content'] for r in results]
        st.session_state.query = query

        # ✅Store in session_state to persist across reruns
        st.session_state.results = results
        st.session_state.resume_texts = [r['content'] for r in results]
        st.session_state.query = query

# ✅ Check if resumes are already retrieved
if "results" in st.session_state and "resume_texts" in st.session_state:
    st.markdown("### Top Matching Resumes ###")

    for i, res in enumerate(st.session_state.results, 1):
        st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; margin-bottom: 20px; border-radius: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);">
        <h4> Resume {i}: <code>{res['filename']}</code></h4>
        <p><strong>✅ ATS Match Score:</strong> <span style="color: green; font-weight: bold;">{res['ats_score']}%</span></p>
        <div style="font-size: 15px; line-height: 1.6; white-space: pre-wrap;">{res['content']}</div>
    </div>
    """, unsafe_allow_html=True)


    # 🧠 LLM Recommendation
    st.markdown("---")
    st.markdown("### LLM Evaluation of Candidates")

    api_key = st.secrets["api"]["pplx_key"]
    st.info("Using secured API key from Streamlit secrets.")

    if st.button(" Analyze with LLM"):
        with st.spinner("LLM is analyzing the candidates..."):
            answer = query_perplexity_llm(
                st.session_state.query,
                st.session_state.resume_texts,
                api_key
            )
        st.success(" LLM has responded!")
        st.markdown("### LLM Recommendation")
        st.markdown(f"""
        <div style="background-color:#e8f5e9;padding:15px;border-radius:10px;">
            {answer}
        </div>
        """, unsafe_allow_html=True)
