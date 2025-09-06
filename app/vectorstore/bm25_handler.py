from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

class BM25Handler:
    def __init__(self, documents, metadata):
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.metadata = metadata
        self.documents = documents

    def search(self, query, top_k=5):
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices based on score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "filename": self.metadata[i],
                "content": self.documents[i],
                "score": scores[i]  # ✅ Include score here
            }
            for i in top_indices
        ]




