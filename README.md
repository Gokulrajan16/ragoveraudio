# 🎙️ Audio RAG System with Voice Query and Response

This application implements a **Retrieval-Augmented Generation (RAG)** system that uses **audio as input and output**, integrating state-of-the-art transcription, vector storage, and language generation.

---

## 🔁 Application Flow

1. **Upload Audio**  
   The user uploads an audio file (e.g., `.wav`, `.mp3`).

2. **Audio Transcription**  
   Audio is transcribed into text using **AssemblyAI**.

3. **Vector Storage in Pinecone**  
   The transcribed text is embedded and stored in a **Pinecone vector database**.

4. **Querying**  
   Users can:
   - Query via **text input**, or  
   - Query using **voice** (speech-to-text).

5. **Answer Generation**  
   The system fetches the most relevant context from Pinecone, and a **Large Language Model (LLM)** (e.g., Sambanova) generates a response.

6. **Audio Response**  
   The generated text answer is **converted to speech** and played back to the user.

---

## 🚀 Features

- 🔉 Upload and transcribe audio
- 🧠 Semantic search with vector database (Pinecone)
- 📣 Query via voice
- 💬 Response generation using LLM
- 🔊 Speak the LLM response (text-to-speech)

---

## 🔐 API Keys Configuration

Create a `.env` file in your project directory with the following content:

```env
PINECONE_API_KEY="your-api-key"
PINECONE_REGION="your-region"
PINECONE_INDEX_NAME="your-index-name"

ASSEMBLYAI_API_KEY="your-api-key"
SAMBANOVA_API_KEY="your-api-key"
