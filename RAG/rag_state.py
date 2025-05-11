# RAG/rag_state.py

from typing import TypedDict, List, Dict, Optional
from langchain_core.documents import Document # If you plan to store lists of Document objects in the state

class OverallState(TypedDict):
   
    # --- Inputs ---
    user_text_query: str              
    # user_audio_path: Optional[str]    
    user_audio_features: Optional[List[float]] # The 94D Essentia features

    # --- Derived from Music Similarity Pipeline & Initial Processing ---
    similar_tracks_summary: Optional[str] # A textual summary of similar tracks found
    project_goal_summary: str           # A synthesized brief combining user query and similar track insights

    # --- Agent-Specific Outputs (Generated Advice) ---
    # These will be populated by their respective agent nodes
    rhythm_advice: Optional[str]
    music_theory_advice: Optional[str]
    instruments_advice: Optional[str]
    lyrics_advice: Optional[str]        # This agent might be conditionally run
    production_advice: Optional[str]

    # --- Final Output ---
    final_moodboard: Optional[str]      # The combined Markdown output

    # --- Control Flags & Error Handling (Optional but useful) ---
    # You could add flags to conditionally run certain agents
    # For example, if the user query doesn't imply lyrics, this could be set to False.
    should_run_lyrics_agent: bool
    error_message: Optional[str]        # To store any error messages encountered during the flow

   