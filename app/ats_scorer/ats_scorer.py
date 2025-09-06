from sentence_transformers import SentenceTransformer, util

# Load MPNet or another sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_ats_score(resume_text: str, job_description: str) -> float:
    """
    Compute ATS score using cosine similarity between resume and JD embeddings.
    Returns score between 0 and 100.
    """
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    
    similarity = util.cos_sim(resume_embedding, jd_embedding).item()
    score = round(similarity * 100, 2)  # Convert to 0–100 scale
    return score
