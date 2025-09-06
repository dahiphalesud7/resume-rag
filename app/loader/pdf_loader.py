
import os
import fitz  # PyMuPDF
import tempfile

def extract_text(pdf_path):
    """Original function for file path"""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc).strip()
    doc.close()
    return text

def load_all_resumes(folder_path):
    """Original function for folder path"""
    texts, filenames = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            texts.append(extract_text(full_path))
            filenames.append(file)
    return filenames, texts

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file object"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = " ".join(page.get_text() for page in doc).strip()
        doc.close()
        return text
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def load_uploaded_resumes(uploaded_files):
    """New function for handling uploaded files"""
    texts, filenames = [], []
    
    for uploaded_file in uploaded_files:
        try:
            # Extract text from uploaded file
            text = extract_text_from_file(uploaded_file)
            texts.append(text)
            filenames.append(uploaded_file.name)
        except Exception as e:
            # Handle errors gracefully
            print(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    return filenames, texts
