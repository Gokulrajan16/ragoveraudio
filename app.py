import os
import gc
import uuid
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from rag_code import Transcribe, EmbedData, Retriever, RAG
import streamlit as st
from pinecone import Pinecone
import assemblyai as aai
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from rag_with_pinecone import PineconeVDB
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import docx
import PyPDF2
from io import BytesIO

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.query = None
    st.session_state.transcripts = []

session_id = st.session_state.id
batch_size = 32

# Load environment variables
env_loaded = load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-multimodal")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.query = None
    gc.collect()

def transcribe_voice(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        fp.write(audio_bytes)
        audio_path = fp.name
    
    try:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speaker_labels=False)
        transcript = transcriber.transcribe(audio_path, config=config)
        return transcript.text if transcript.text else None
    finally:
        os.unlink(audio_path)

def extract_text_from_file(uploaded_file):
    text = ""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type in ["mp3", "wav", "m4a"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                transcriber = Transcribe(api_key=ASSEMBLYAI_API_KEY)
                transcripts = transcriber.transcribe_audio(file_path)
                st.session_state.transcripts = transcripts
                text = " ".join([f"Speaker {t['speaker']}: {t['text']}" for t in transcripts])
        
        elif file_type == "pdf":
            # Try text extraction first
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            
            # If text extraction fails (scanned PDF), use OCR
            if not text.strip():
                images = convert_from_bytes(uploaded_file.getvalue())
                text = "\n".join([pytesseract.image_to_string(image) for image in images])
        
        elif file_type == "docx":
            doc = docx.Document(BytesIO(uploaded_file.getvalue()))
            text = "\n".join([para.text for para in doc.paragraphs])
        
        elif file_type == "txt":
            text = uploaded_file.getvalue().decode("utf-8")
        
        elif file_type in ["png", "jpg", "jpeg"]:
            image = Image.open(BytesIO(uploaded_file.getvalue()))
            text = pytesseract.image_to_string(image)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None
    
    return text

# Sidebar for file upload
with st.sidebar:
    st.header("Add your files!")
    uploaded_file = st.file_uploader("Choose your file", 
                                   type=["mp3", "wav", "m4a", "pdf", "docx", "txt", "png", "jpg", "jpeg"])

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"
            st.write("Processing file...")
            
            if file_key not in st.session_state.get('file_cache', {}):
                extracted_text = extract_text_from_file(uploaded_file)
                
                if not extracted_text:
                    st.error("Could not extract text from the file")
                    st.stop()
                
                # Split text into chunks
                chunk_size = 1000  # characters
                documents = [extracted_text[i:i+chunk_size] 
                            for i in range(0, len(extracted_text), chunk_size)]
                
                embeddata = EmbedData(embed_model_name="BAAI/bge-large-en-v1.5", batch_size=batch_size)
                embeddata.embed(documents)
                
                pinecone_vdb = PineconeVDB(index_name=PINECONE_INDEX_NAME, 
                                          embed_dim=len(embeddata.embeddings[0]))
                pinecone_vdb.ingest_data(embeddata=embeddata)
                
                retriever = Retriever(vector_db=pinecone_vdb, embeddata=embeddata)
                query_engine = RAG(retriever=retriever, llm_name="DeepSeek-R1-Distill-Llama-70B")
                st.session_state.file_cache[file_key] = query_engine
            
            st.success("Ready to Chat!")
            
            # Show appropriate preview
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext in ["mp3", "wav", "m4a"]:
                st.audio(uploaded_file)
                with st.expander("View Transcript"):
                    if 'transcripts' in st.session_state:
                        for t in st.session_state.transcripts:
                            st.write(f"**{t['speaker']}**: {t['text']}")
            elif file_ext in ["png", "jpg", "jpeg"]:
                st.image(uploaded_file, caption="Uploaded Image")
            elif file_ext == "pdf":
                st.write("PDF document uploaded")
                with st.expander("View Extracted Text"):
                    st.text(extracted_text[:5000] + ("..." if len(extracted_text) > 5000 else ""))
            elif file_ext == "docx":
                st.write("Word document uploaded")
                with st.expander("View Extracted Text"):
                    st.text(extracted_text[:5000] + ("..." if len(extracted_text) > 5000 else ""))
            elif file_ext == "txt":
                st.write("Text file uploaded")
                with st.expander("View Content"):
                    st.text(extracted_text[:5000] + ("..." if len(extracted_text) > 5000 else ""))
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Main interface
st.title("Multimodal RAG Assistant")
st.button("Clear Chat", on_click=reset_chat)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Voice & Text Input
st.subheader("Ask a Question")
voice_tab, text_tab = st.tabs(["🎤 Voice", "✏️ Text"])

with voice_tab:
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
    )
    if audio_bytes:
        with st.spinner("Transcribing..."):
            transcribed_text = transcribe_voice(audio_bytes)
            if transcribed_text:
                st.text_area("You said:", value=transcribed_text, height=100)
                st.session_state.query = transcribed_text
                if st.button("Submit Voice Query"):
                    st.session_state.trigger_query = True

with text_tab:
    typed_text = st.chat_input("Type your question...")
    if typed_text:
        st.session_state.query = typed_text
        st.session_state.trigger_query = True

# Process query
if st.session_state.get("trigger_query") and st.session_state.query:
    query = st.session_state.query
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if 'file_cache' in st.session_state and st.session_state.file_cache:
            query_engine = next(iter(st.session_state.file_cache.values()))
            response = query_engine.query(query)

            message_placeholder = st.empty()
            full_response = ""

            for chunk in response:
                try:
                    full_response += chunk.raw["choices"][0]["delta"]["content"]
                    message_placeholder.markdown(full_response + "▌")
                except:
                    pass

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Text-to-Speech
            tts = gTTS(text=full_response)
            audio_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.mp3"
            tts.save(audio_path)

            audio_bytes = open(audio_path, 'rb').read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning("Please upload a file first")

    st.session_state.query = None
    st.session_state.trigger_query = False