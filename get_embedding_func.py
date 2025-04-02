from get_embedding_func import SentenceTransformerEmbeddings

# def get_embedding_function():
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size  # Control batch size

    def embed_documents(self, texts):
        texts = ["search_document: "+i for i in texts]
        return self.model.encode(
            texts, 
            convert_to_numpy=True, 
            device=self.device, 
            batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            ['search_query: '+text], 
            convert_to_numpy=True, 
            device=self.device
        )[0].tolist()