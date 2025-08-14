from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, doc_path):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = open(doc_path).read().split("\n")
        self.embeddings = self.model.encode(self.docs)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def retrieve(self, query, top_k=1):
        query_vec = self.model.encode([query])
        _, indices = self.index.search(query_vec, top_k)
        return [self.docs[i] for i in indices[0]]

    def generate(self, input_text, retrieved_docs):
        # Simple concatenation for demo
        context = " ".join(retrieved_docs)
        return f"Input: {input_text}\nContext: {context}\nResponse: This is a simulated answer based on both."