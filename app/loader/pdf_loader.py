import os
import fitz  # PyMuPDF

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join(page.get_text() for page in doc).strip()

def load_all_resumes(folder_path):
    texts, filenames = [], []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            texts.append(extract_text(full_path))
            filenames.append(file)
    return filenames, texts

