import streamlit as st
import os
import tempfile
import fitz
import nltk
nltk.download('punkt_tab')


from app.loader.pdf_loader import load_all_resumes, load_uploaded_resumes
from app.processor.cleaner import clean_text
from app.embedder.text_embedder import get_embeddings
from app.vectorstore.faiss_handler import FaissHandler
from app.vectorstore.bm25_handler import BM25Handler
from app.retriever.hybrid_retriever import HybridRetriever
from app.llm.perplexity_llm import query_perplexity_llm
from app.ats_scorer.ats_scorer import compute_ats_score

# Temporary function in case import fails
def load_uploaded_resumes_temp(uploaded_files):
    """Temporary function for handling uploaded files"""
    texts, filenames = [], []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            # Extract text using PyMuPDF
            doc = fitz.open(tmp_path)
            text = " ".join(page.get_text() for page in doc).strip()
            doc.close()
            
            texts.append(text)
            filenames.append(uploaded_file.name)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return filenames, texts

# 🎯 Page setup
st.set_page_config(page_title="Resume Retriever", layout="wide")
st.title("Resume Retrieval System (BM25 + FAISS Hybrid + Perplexity LLM)")

# 🔄 Session state initialization
for key in ["texts", "metadata", "embeddings", "bm25", "faiss"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================== CHOOSE INPUT METHOD ==================
st.markdown("Choose Input Method")

input_method = st.radio(
    "How would you like to provide resumes?",
    ["Upload PDF Files", "Use Folder Path"],
    horizontal=True
)

# ================== METHOD 1: FILE UPLOAD ==================
if input_method == "Upload PDF Files":
    st.markdown("Upload Resume Files")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select multiple PDF resume files to analyze",
        key="pdf_uploader"
    )
    
    # Debug info
    if st.checkbox("Show Debug Info"):
        st.write("**Debug Information:**")
        st.write("uploaded_files:", uploaded_files)
        st.write("Type:", type(uploaded_files))
        if uploaded_files:
            st.write("Number of files:", len(uploaded_files))
            for i, file in enumerate(uploaded_files):
                st.write(f"File {i+1}: {file.name} ({file.size} bytes)")
    
    # Show uploaded files
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected successfully!")
        
        with st.expander("Uploaded Files", expanded=True):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. **{file.name}** - {file.size:,} bytes")
    
    # Process uploaded files
    if uploaded_files and st.button("Process Uploaded Resumes", key="process_uploads"):
        with st.spinner("Processing uploaded resumes..."):
            try:
                st.info(f"Starting to process {len(uploaded_files)} files...")
                
                # Try to use the imported function, fall back to temp function
                try:
                    metadata, docs = load_uploaded_resumes(uploaded_files)
                except:
                    st.warning("Using temporary upload function...")
                    metadata, docs = load_uploaded_resumes_temp(uploaded_files)
                
                if not docs:
                    st.error("No valid PDF files found. Please check your uploads.")
                else:
                    st.info("Cleaning text...")
                    cleaned_texts = [clean_text(doc) for doc in docs]

                    st.info("Storing processed data...")
                    st.session_state.texts = cleaned_texts
                    st.session_state.metadata = metadata

                    st.info("Initializing BM25 handler...")
                    st.session_state.bm25 = BM25Handler(cleaned_texts, metadata)
                    
                    st.info("Generating embeddings...")
                    st.session_state.embeddings = get_embeddings(cleaned_texts)
                    
                    st.info("Initializing FAISS handler...")
                    st.session_state.faiss = FaissHandler(cleaned_texts, metadata)

                    st.success(f"Successfully processed {len(docs)} resumes!")
                    
                    with st.expander("Processing Summary", expanded=True):
                        st.write(f"**Total Resumes:** {len(docs)}")
                        st.write(f"**Processed Files:**")
                        for i, filename in enumerate(metadata, 1):
                            st.write(f"  {i}. {filename}")
                    
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

# ================== METHOD 2: FOLDER PATH ==================
elif input_method == "Use Folder Path":
    st.markdown("#### Folder Path Input")
    
    folder_path = st.text_input("Enter folder path with PDF resumes:", "resumes")
    
    # Show current directory info
    with st.expander("Directory Information"):
        st.write(f"**Current Directory:** {os.getcwd()}")
        try:
            current_items = os.listdir(".")
            folders = [item for item in current_items if os.path.isdir(item)]
            files = [item for item in current_items if os.path.isfile(item)]
            
            st.write(f"**Available Folders:** {folders if folders else 'None'}")
            st.write(f"**Available Files:** {len(files)} files")
        except Exception as e:
            st.write(f"Cannot list directory: {e}")
    
    if st.button("Load and Process Resumes", key="process_folder"):
        if not os.path.exists(folder_path):
            st.error(f"Folder path '{folder_path}' does not exist.")
        else:
            with st.spinner("Processing folder resumes..."):
                try:
                    metadata, docs = load_all_resumes(folder_path)
                    
                    if not docs:
                        st.error(f"No PDF files found in '{folder_path}' folder.")
                    else:
                        cleaned_texts = [clean_text(doc) for doc in docs]

                        st.session_state.texts = cleaned_texts
                        st.session_state.metadata = metadata

                        st.session_state.bm25 = BM25Handler(cleaned_texts, metadata)
                        st.session_state.embeddings = get_embeddings(cleaned_texts)
                        st.session_state.faiss = FaissHandler(cleaned_texts, metadata)

                        st.success(f"Successfully processed {len(docs)} resumes from folder!")
                        
                        with st.expander("Processing Summary"):
                            st.write(f"**Total Resumes:** {len(docs)}")
                            st.write(f"**Processed Files:** {', '.join(metadata)}")
                        
                except Exception as e:
                    st.error(f"Error processing folder: {str(e)}")
                    st.exception(e)

# ================== SEARCH INTERFACE ==================
st.markdown("---")
st.markdown("Search Interface")

# Show processing status
if st.session_state.texts is not None:
    st.info(f"{len(st.session_state.texts)} resumes loaded and ready for search!")
else:
    st.warning("No resumes loaded. Please upload files or specify a folder path above.")

query = st.text_input("Enter job description or query:", 
                     placeholder="e.g., Python developer with machine learning experience")

if query and st.button("Retrieve Matching Resumes"):
    if st.session_state.texts is None or st.session_state.embeddings is None:
        st.warning("Please process resumes before searching.")
    else:
        with st.spinner("Searching for matching resumes..."):
            try:
                hybrid = HybridRetriever(
                    bm25_handler=st.session_state.bm25,
                    faiss_handler=st.session_state.faiss
                )
                results = hybrid.retrieve(query)
                
                # Calculate ATS scores
                for res in results:
                    res["ats_score"] = compute_ats_score(res["content"], query)
                
                # Sort by ATS score
                results.sort(key=lambda x: x['ats_score'], reverse=True)

                # Store in session_state
                st.session_state.results = results
                st.session_state.resume_texts = [r['content'] for r in results]
                st.session_state.query = query
                
                st.success(f"Found {len(results)} matching resumes!")
                
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
                st.exception(e)

# ================== DISPLAY RESULTS ==================
if "results" in st.session_state and st.session_state.results:
    st.markdown("---")
    st.markdown("Top Matching Resumes")

    for i, res in enumerate(st.session_state.results, 1):
        with st.expander(f"Resume {i}: {res['filename']} (ATS Score: {res['ats_score']}%)", 
                        expanded=(i <= 3)):  # Expand first 3 results
            
            # Score visualization
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("ATS Score", f"{res['ats_score']}%")
            with col2:
                # Simple progress bar
                progress = res['ats_score'] / 100
                st.progress(progress)
            
            # Resume content
            st.markdown("**Resume Content:**")
            st.text_area("", res['content'], height=200, key=f"resume_{i}")

    # ================== LLM ANALYSIS ==================
    st.markdown("---")
    st.markdown("AI Analysis")

    try:
        api_key = st.secrets["api"]["pplx_key"]
        st.info("API key configured successfully")
        
        if st.button("Analyze with AI", key="llm_analyze"):
            with st.spinner("AI is analyzing the candidates..."):
                try:
                    answer = query_perplexity_llm(
                        st.session_state.query,
                        st.session_state.resume_texts,
                        api_key
                    )
                    
                    st.success("AI analysis completed!")
                    
                    with st.container():
                        st.markdown("AI Recommendation")
                        st.markdown(f"""
                        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px;border-left:5px solid #4caf50;">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error during AI analysis: {str(e)}")
                    
    except Exception as e:
        st.warning("API key not configured. Please add your Perplexity API key in Streamlit secrets.")
        st.info("To add API key: App Settings → Secrets → Add `[api]` and `pplx_key = 'your_key'`")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("Instructions")
with st.expander("How to use this app", expanded=False):
    st.markdown("""
    1. **Upload PDFs** or specify a folder path with resume files
    2. **Wait for processing** - the system will extract text and create embeddings
    3. **Enter a job description** or search query
    4. **View ranked results** based on ATS scores
    5. **Get AI analysis** for deeper insights (requires API key)
    
    **Supported formats:** PDF files only
    **File size limit:** 200MB per file
    """)