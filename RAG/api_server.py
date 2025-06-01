# RAG/api_server.py

import logging
import os  # Add this import
from typing import Dict, Any, Optional, List
import sys

from fastapi import FastAPI, HTTPException, Body # Body might not be needed if using Pydantic model
from pydantic import BaseModel, Field
import uvicorn

from rag_state import OverallState
from rag_music_moodboard import build_graph # Import the function that builds the graph
# Importing the configuration for logging and LLM model name
from rag_config import LLM_MODEL_NAME # For logging startup info



# --- Setup Logger ---
def setup_console_logging():
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)
    
    # Also configure specific loggers
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    return logger

logger = setup_console_logging()
logger.info("RAG API Server logging configured with console output")


# --- Pydantic Models for API Request and Response ---
class MoodboardRequest(BaseModel):
    user_text_query: str = Field(..., description="The user's natural language query for music inspiration.")
    similar_tracks_summary: str = Field(..., description="A summary (e.g., Markdown table text) of acoustically similar tracks or general context if no audio was analyzed.")
 

class MoodboardResponse(BaseModel):
    final_moodboard: Optional[str] = Field(None, description="The generated music inspiration moodboard in Markdown format.")
    all_accumulated_sources: Optional[List[str]] = Field(None, description="List of sources referenced.")
    total_prompt_tokens: Optional[int] = Field(None, description="Total prompt tokens used for this request.")
    total_completion_tokens: Optional[int] = Field(None, description="Total completion tokens used for this request.")
    error_message: Optional[str] = Field(None, description="Any error message encountered during processing.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Music Inspiration Moodboard RAG API",
    description="Generates a music inspiration moodboard using a RAG system based on text query and track similarity context.",
    version="0.1.0"
)

# --- Compile LangGraph App ---
moodboard_graph_app = None
try:
    # Assuming concurrent mode is default for API
    moodboard_graph_app = build_graph(sequential_mode=False).compile()
    logger.info("LangGraph application compiled successfully for API server.")
except Exception as e:
    logger.error(f"FATAL: Could not compile LangGraph application: {e}", exc_info=True)


# --- API Endpoints ---
@app.post("/generate_moodboard", response_model=MoodboardResponse)
async def generate_moodboard_endpoint(request_data: MoodboardRequest):
    """
    Generates a music inspiration moodboard based on user query and similar track analysis.
    """
    logger.info(f"Received request for /generate_moodboard. User query: '{request_data.user_text_query[:50]}...'")
    logger.debug(f"Full user query: {request_data.user_text_query}")  # Add this
    logger.debug(f"Full similar_tracks_summary: {request_data.similar_tracks_summary}")  # Add this


    if not moodboard_graph_app:
        logger.error("LangGraph application is not compiled. Cannot process request.")
        raise HTTPException(status_code=503, detail="RAG system not ready or compilation failed.") # 503 Service Unavailable

    # 1. Prepare Initial State for the Graph
    initial_graph_state = OverallState(
        user_text_query=request_data.user_text_query,
        user_audio_path=None, # Explicitly None as it's removed from API request
        user_audio_features=None,
        similar_tracks_summary=request_data.similar_tracks_summary,
        project_goal_summary="",
        
        rhythm_advice=None, rhythm_kb_sources=[], rhythm_stack_sources=[],
        music_theory_advice=None, music_theory_kb_sources=[], music_theory_stack_sources=[],
        instruments_advice=None, instruments_kb_sources=[], instruments_stack_sources=[],
        lyrics_advice=None, lyrics_kb_sources=[], lyrics_stack_sources=[],
        production_advice=None, production_kb_sources=[], production_stack_sources=[],
        
        all_accumulated_sources=[],
        final_moodboard=None,
        error_message=None,
        should_run_lyrics_agent=False,

        node_prompt_tokens=0, node_completion_tokens=0,
        rhythm_se_query_prompt_tokens=0, rhythm_se_query_completion_tokens=0,
        rhythm_final_advice_prompt_tokens=0, rhythm_final_advice_completion_tokens=0,
        music_theory_se_query_prompt_tokens=0, music_theory_se_query_completion_tokens=0,
        music_theory_final_advice_prompt_tokens=0, music_theory_final_advice_completion_tokens=0,
        instruments_se_query_prompt_tokens=0, instruments_se_query_completion_tokens=0,
        instruments_final_advice_prompt_tokens=0, instruments_final_advice_completion_tokens=0,
        lyrics_se_query_prompt_tokens=0, lyrics_se_query_completion_tokens=0,
        lyrics_final_advice_prompt_tokens=0, lyrics_final_advice_completion_tokens=0,
        production_se_query_prompt_tokens=0, production_se_query_completion_tokens=0,
        production_final_advice_prompt_tokens=0, production_final_advice_completion_tokens=0,
        total_prompt_tokens=0, total_completion_tokens=0, total_cost=0.0,
    )
    logger.info(f"Prepared initial state for LangGraph invocation.")
    logger.debug(f"Full initial state for graph (user query only): {initial_graph_state['user_text_query']}")

    final_result_state: Optional[OverallState] = None
    try:
        logger.info("Invoking LangGraph app...")
        logger.debug(f"Initial graph state: {initial_graph_state}")  # Add this
        final_result_state = moodboard_graph_app.invoke(initial_graph_state)
        logger.debug(f"Graph invocation result: {final_result_state}")  # Add this
        logger.info("LangGraph app invocation complete.")
        if final_result_state:
            logger.debug(f"Final result state from graph (keys): {list(final_result_state.keys())}")
        else:
            logger.warning("Graph invocation returned None for final_result_state.")

    except Exception as e:
        logger.error(f"Error invoking LangGraph app: {e}", exc_info=True)
        # Consider what to return here, maybe the error from final_result_state if it got that far
        # or a generic error.
        error_detail = final_result_state.get("error_message", str(e)) if final_result_state else str(e)
        raise HTTPException(status_code=500, detail=f"Error generating moodboard: {error_detail}")

    if not final_result_state:
        logger.error("Moodboard generation failed: No final state returned from graph invocation.")
        raise HTTPException(status_code=500, detail="Moodboard generation failed: No final state from RAG system.")

    # 3. Prepare and Return Response
    return MoodboardResponse(
        final_moodboard=final_result_state.get("final_moodboard"),
        all_accumulated_sources=final_result_state.get("all_accumulated_sources", []),
        total_prompt_tokens=final_result_state.get("total_prompt_tokens"),
        total_completion_tokens=final_result_state.get("total_completion_tokens"),
        error_message=final_result_state.get("error_message")
    )

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # This assumes that the main entry point for logging (e.g. in run_music_moodboard.py via setup_logging)
    # has already configured the root logger if you run this directly.
    # If you run `python -m RAG.api_server`, the logger instance here will pick up
    # whatever root configuration is active.
    if not logging.getLogger().handlers: # Ensure at least one handler if running standalone
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        logger.warning("Configured basic logging for standalone api_server.py run.")

    logger.info("Starting RAG API server with Uvicorn on 0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") # Uvicorn has its own logging format for access logs