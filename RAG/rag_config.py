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
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "pdf_knowledge_base_hybrid3")

# --- Embedding Model Configuration ---
DENSE_EMBEDDING_MODEL_NAME = os.getenv("DENSE_EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
SPARSE_EMBEDDING_MODEL_NAME = os.getenv("SPARSE_EMBEDDING_MODEL_NAME", "Qdrant/bm25")

# --- LLM Configuration ---
LLM_PROVIDER = "LocalOpenAICompatible" 

# Local LLM Setup (Jan, Ollama, etc.)
LOCAL_LLM_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:1337/v1")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "llama3.2:3b")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "NotNeeded")

# Remote LLM Setup (RunPod, OpenAI-compatible APIs)
REMOTE_LLM_API_BASE = os.getenv("REMOTE_LLM_API_BASE", "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1")
REMOTE_LLM_MODEL_NAME = os.getenv("REMOTE_LLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
REMOTE_LLM_API_KEY = os.getenv("REMOTE_LLM_API_KEY")

# Active LLM Configuration (the ones actually used by the system)
LLM_API_BASE = os.getenv("LLM_API_BASE", REMOTE_LLM_API_BASE)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", REMOTE_LLM_MODEL_NAME)
LLM_API_KEY = os.getenv("LLM_API_KEY", REMOTE_LLM_API_KEY)

# --- Retrieval Configuration ---
DEFAULT_RETRIEVAL_TOP_K = int(os.getenv("DEFAULT_RETRIEVAL_TOP_K", 3))

# --- External APIs ---
STACKEXCHANGE_API_KEY = os.getenv("STACKEXCHANGE_API_KEY")

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG_PROMPTS = os.getenv("DEBUG_PROMPTS", "false").lower() == "true"

# --- Development/Testing Configuration ---
DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", 30))
API_REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", 10))

# --- Agent Configuration ---
AGENT_TOPICS = ["rythm", "production", "genre_styles", "theory_general", "timbre_instruments"]

if __name__ == '__main__':
    print("--- RAG Configuration Summary ---")
    print(f"QDRANT_URL: {QDRANT_URL}")
    print(f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}")
    print(f"DENSE_EMBEDDING_MODEL_NAME: {DENSE_EMBEDDING_MODEL_NAME}")
    print(f"SPARSE_EMBEDDING_MODEL_NAME: {SPARSE_EMBEDDING_MODEL_NAME}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"")
    print("--- Active LLM Configuration ---")
    print(f"LLM_API_BASE: {LLM_API_BASE}")
    print(f"LLM_MODEL_NAME: {LLM_MODEL_NAME}")
    print(f"LLM_API_KEY: {'***' + LLM_API_KEY[-8:] if LLM_API_KEY and len(LLM_API_KEY) > 8 else 'Not Set'}")
    print(f"")
    print("--- Local LLM Configuration ---")
    print(f"LOCAL_LLM_API_BASE: {LOCAL_LLM_API_BASE}")
    print(f"LOCAL_LLM_MODEL_NAME: {LOCAL_LLM_MODEL_NAME}")
    print(f"")
    print("--- Remote LLM Configuration ---")
    print(f"REMOTE_LLM_API_BASE: {REMOTE_LLM_API_BASE}")
    print(f"REMOTE_LLM_MODEL_NAME: {REMOTE_LLM_MODEL_NAME}")
    print(f"")
    print("--- Other Settings ---")
    print(f"DEFAULT_RETRIEVAL_TOP_K: {DEFAULT_RETRIEVAL_TOP_K}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
    print(f"DEBUG_PROMPTS: {DEBUG_PROMPTS}")
    print(f"DEVELOPMENT_MODE: {DEVELOPMENT_MODE}")
    print(f"LLM_REQUEST_TIMEOUT: {LLM_REQUEST_TIMEOUT}s")
    print(f"API_REQUEST_TIMEOUT: {API_REQUEST_TIMEOUT}s")
    print(f"STACKEXCHANGE_API_KEY: {'Set' if STACKEXCHANGE_API_KEY else 'Not Set'}")
    print(f"AGENT_TOPICS: {AGENT_TOPICS}")
    print(f"LOCAL_LLM_API_KEY: {LOCAL_LLM_API_KEY}")
    print(f"DEFAULT_RETRIEVAL_TOP_K: {DEFAULT_RETRIEVAL_TOP_K}")
    print(f"AGENT_TOPICS: {AGENT_TOPICS}")