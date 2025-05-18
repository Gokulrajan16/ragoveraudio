import os
from typing import List, Dict
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import re

def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

class EmbedData:
    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        return HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            trust_remote_code=True,
            cache_folder='./hf_cache'
        )

    def generate_embedding(self, context):
        return self.embed_model.get_text_embedding_batch(context)

    def embed(self, contexts):
        self.contexts = contexts
        for batch_context in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query, top_k=5):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        response = self.vector_db.query(query, top_k=top_k)
        return response.matches

class RAG:
    def __init__(self, retriever, llm_name="Meta-Llama-3.1-405B-Instruct"):
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions about the user's document."
        )
        self.messages = [system_msg]
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.qa_prompt_tmpl_str = (
            """Context information is below.
            ---------------------
            {context}
            ---------------------
            Given the context information and no prior knowledge, answer the query.
            Query: {query}
            Answer: """
        )

    def _setup_llm(self):
        return SambaNovaCloud(
            model=self.llm_name,
            temperature=0.7,
            context_window=100000
        )

    def generate_context(self, query):
        result = self.retriever.search(query)
        combined_prompt = []
        for entry in result[:3]:  # Get top 3 results
            context = entry.metadata["text"]
            combined_prompt.append(context)
        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query):
        context = self.generate_context(query=query)
        prompt = self.qa_prompt_tmpl_str.format(context=context, query=query)
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        streaming_response = self.llm.stream_complete(user_msg.content)
        return streaming_response

class Transcribe:
    def __init__(self, api_key: str):
        import assemblyai as aai
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()

    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        import assemblyai as aai
        config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=2)
        transcript = self.transcriber.transcribe(audio_path, config=config)
        return [{"speaker": f"Speaker {u.speaker}", "text": u.text} for u in transcript.utterances]