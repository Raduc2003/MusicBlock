# RAG/test_llm_connection.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load configurations directly for this test, mimicking rag_config.py
load_dotenv() # Assumes .env file is in project root or RAG/ directory

# --- Configuration (copy relevant parts from your rag_config.py) ---
LLM_PROVIDER = "LocalOpenAICompatible" # As per your choice

# For Jan (or other OpenAI-compatible local server)
LOCAL_LLM_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:1337/v1")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "bartowski:Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q8_0.gguf") # **IMPORTANT: SET THIS IN .ENV or here**
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "NotNeeded")

print("--- LLM Test Configuration ---")
print(f"LLM_PROVIDER: {LLM_PROVIDER}")
print(f"LOCAL_LLM_API_BASE: {LOCAL_LLM_API_BASE}")
print(f"LOCAL_LLM_MODEL_NAME: {LOCAL_LLM_MODEL_NAME}") # Make sure this prints the correct model name
print(f"LOCAL_LLM_API_KEY: {LOCAL_LLM_API_KEY}") # See what this is resolving to

llm = None
try:
    if LLM_PROVIDER == "LocalOpenAICompatible":
        api_key_to_use = None # Default to None
        if LOCAL_LLM_API_KEY and LOCAL_LLM_API_KEY.lower() not in ["notneeded", "none", ""]:
            api_key_to_use = LOCAL_LLM_API_KEY
        # If your Jan server strictly needs some string, even if it's ignored:
        elif LOCAL_LLM_API_KEY and LOCAL_LLM_API_KEY.lower() in ["notneeded", "none", ""]:
             api_key_to_use = "NotNeeded" # Or try an empty string "" or "dummy"

        print(f"Attempting to initialize ChatOpenAI with API Key: '{api_key_to_use}'")

        llm = ChatOpenAI(
            model=LOCAL_LLM_MODEL_NAME,
            openai_api_base=LOCAL_LLM_API_BASE,
            api_key=api_key_to_use,
            temperature=0.7,
            # Explicitly set other parameters that OpenAI client might default
            # stream=False, # invoke usually doesn't stream by default, but let's be sure
            # max_tokens=500, # Set a reasonable max_tokens
            default_headers={"Content-Type": "application/json"} # Usually not needed as client handles it
        )
        print(f"\nSUCCESS: LLM object initialized: {llm}")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

except Exception as e:
    print(f"\nERROR during LLM initialization: {e}")
    import traceback
    traceback.print_exc()

if llm:
    print("\n--- Attempting a simple LLM call ---")
    try:
        messages = [{"role": "user", "content": "TElll a joke"}] # Same prompt
        print(f"Sending messages: {messages}")
        
        # Using invoke, but can pass model_kwargs if necessary
        # For ChatOpenAI, temperature is a direct param, not model_kwargs
        response = llm.invoke(messages) # temperature is already set on the llm object

        print("\nSUCCESS: LLM Response:")
        if hasattr(response, 'content'):
            print(response.content)
        else:
            print(response)
    except Exception as e:
        print(f"\nERROR during LLM invoke: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nSkipping LLM call because initialization failed.")