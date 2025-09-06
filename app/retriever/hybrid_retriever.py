# app/retriever/hybrid_retriever.py

class HybridRetriever:
    def __init__(self, faiss_handler, bm25_handler, alpha=0.5):
        self.faiss = faiss_handler
        self.bm25 = bm25_handler
        self.alpha = alpha

    def retrieve(self, query, top_k=5):
        faiss_results = self.faiss.search(query, top_k=top_k)
        bm25_results = self.bm25.search(query, top_k=top_k)

        faiss_dict = {r["filename"]: r["score"] for r in faiss_results}
        bm25_dict = {r["filename"]: r["score"] for r in bm25_results}
        content_dict = {r["filename"]: r.get("content", "") for r in bm25_results}

        all_filenames = set(faiss_dict) | set(bm25_dict)

        combined = []
        for fname in all_filenames:
            faiss_score = faiss_dict.get(fname, 0)
            bm25_score = bm25_dict.get(fname, 0)
            final_score = self.alpha * faiss_score + (1 - self.alpha) * bm25_score
            combined.append({
                "filename": fname,
                "content": content_dict.get(fname, ""),  # prefer full BM25 content
                "score": final_score
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

