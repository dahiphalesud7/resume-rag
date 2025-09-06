from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class FaissHandler:
    def __init__(self, documents, metadata):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents
        self.metadata = metadata

        # Generate embeddings
        self.embeddings = self.model.encode(documents, show_progress_bar=True)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])

        # Add embeddings to index
        self.index.add(np.array(self.embeddings).astype("float32"))

    def search(self, query, top_k=5):
        # Embed the query
        query_vec = self.model.encode([query])[0].astype("float32")

        # Search in the index
        distances, indices = self.index.search(np.array([query_vec]), top_k)

        # Convert distances to similarity scores
        results = []
        for i, idx in enumerate(indices[0]):
            distance = distances[0][i]
            similarity = 1 / (1 + distance)  # higher = more similar

            results.append({
                "filename": self.metadata[idx],
                "score": similarity,
                "content": self.documents[idx]
            })

        return results
