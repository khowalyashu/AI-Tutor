from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        if not texts:
            return []

        texts = [f"search_document: {text.strip()}" for text in texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            device=self.device,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text):
        text = f"search_query: {text.strip()}"
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            device=self.device,
            show_progress_bar=False
        )
        return embedding.tolist()

def get_embedding_function():
    return SentenceTransformerEmbeddings()
