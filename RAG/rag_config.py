# RAG/rag_config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "pdf_knowledge_base_hybrid3"

# --- Embedding Model Configuration ---
DENSE_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
SPARSE_EMBEDDING_MODEL_NAME = "Qdrant/bm25"

# --- LLM Configuration ---
LLM_PROVIDER = "LocalOpenAICompatible" 


LOCAL_LLM_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:1337/v1") # Jan default, adjust if needed. Ollama uses /v1 too.

LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "bartowski:Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q8_0.gguf") # e.g., "mistral-7b-instruct-v0.2-q4_K_M" if that's what Jan shows
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "NotNeeded") # Often not needed for local servers, but good to have a placeholder

# --- Retrieval Configuration ---
DEFAULT_RETRIEVAL_TOP_K = 3

# --- Agent Configuration ---
AGENT_TOPICS = ["rythm", "production", "genre_styles", "theory_general", "timbre_instruments"]

# --- Paths (Optional) ---
# PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

if __name__ == '__main__':
    print("--- RAG Configurations ---")
    print(f"QDRANT_URL: {QDRANT_URL}")
    print(f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}")
    print(f"DENSE_EMBEDDING_MODEL_NAME: {DENSE_EMBEDDING_MODEL_NAME}")
    print(f"SPARSE_EMBEDDING_MODEL_NAME: {SPARSE_EMBEDDING_MODEL_NAME}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"LOCAL_LLM_API_BASE: {LOCAL_LLM_API_BASE}")
    print(f"LOCAL_LLM_MODEL_NAME: {LOCAL_LLM_MODEL_NAME}")
    print(f"LOCAL_LLM_API_KEY: {LOCAL_LLM_API_KEY}")
    print(f"DEFAULT_RETRIEVAL_TOP_K: {DEFAULT_RETRIEVAL_TOP_K}")
    print(f"AGENT_TOPICS: {AGENT_TOPICS}")