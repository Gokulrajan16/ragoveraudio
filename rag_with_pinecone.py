from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

class PineconeVDB:
    def __init__(self, index_name: str, embed_dim: int = 1024):
        self.index_name = index_name
        self.embed_dim = embed_dim
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True,
            cache_folder='./hf_cache'
        )
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Check and create index if needed
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embed_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)

    def ingest_data(self, embeddata):
        """Ingest embeddings and metadata into Pinecone"""
        vectors = []
        for idx, (context, embedding) in enumerate(zip(embeddata.contexts, embeddata.embeddings)):
            vectors.append({
                "id": f"vec-{idx}",
                "values": embedding,
                "metadata": {"text": context}
            })
        
        # Upsert in batches of 100 (Pinecone limit)
        for batch in self._chunkify(vectors, 100):
            self.index.upsert(vectors=batch)

    def query(self, query: str, top_k: int = 5):
        """Query the index"""
        query_embedding = self.embed_model.get_query_embedding(query)
        return self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

    @staticmethod
    def _chunkify(lst, chunk_size):
        """Yield successive chunk_size chunks from lst"""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]