# RAG/rag_state.py

from typing import TypedDict, List, Dict, Optional
from langchain_core.documents import Document # If you plan to store lists of Document objects in the state

class OverallState(TypedDict):
   
    # --- Inputs ---
    user_text_query: str              
    user_audio_path: Optional[str]    
    user_audio_features: Optional[List[float]] # The 94D Essentia features

    # --- Derived from Music Similarity Pipeline & Initial Processing ---
    similar_tracks_summary: Optional[str] # A textual summary of similar tracks found
    project_goal_summary: str           # A synthesized brief combining user query and similar track insights

    # --- Agent-Specific Outputs (Generated Advice) ---
    rhythm_advice: Optional[str]
    rhythm_kb_sources: Optional[List[str]]
    rhythm_stack_sources: Optional[List[str]]

    music_theory_advice: Optional[str]
    music_theory_kb_sources: Optional[List[str]]
    music_theory_stack_sources: Optional[List[str]]

    instruments_advice: Optional[str]
    instruments_kb_sources: Optional[List[str]]
    instruments_stack_sources: Optional[List[str]]

    lyrics_advice: Optional[str]
    lyrics_kb_sources: Optional[List[str]]
    lyrics_stack_sources: Optional[List[str]]

    production_advice: Optional[str]
    production_kb_sources: Optional[List[str]]
    production_stack_sources: Optional[List[str]]
    
    all_accumulated_sources: Optional[List[str]] # For the combiner to output

    # --- Token Tracking ---
    # Node-level token counters (for current processing node)
    node_prompt_tokens: int
    node_completion_tokens: int
    
    # Agent-specific token counters for SE query generation
    rhythm_se_query_prompt_tokens: int
    rhythm_se_query_completion_tokens: int
    music_theory_se_query_prompt_tokens: int
    music_theory_se_query_completion_tokens: int
    instruments_se_query_prompt_tokens: int
    instruments_se_query_completion_tokens: int
    lyrics_se_query_prompt_tokens: int
    lyrics_se_query_completion_tokens: int
    production_se_query_prompt_tokens: int
    production_se_query_completion_tokens: int
    
    # Agent-specific token counters for final advice generation
    rhythm_final_advice_prompt_tokens: int
    rhythm_final_advice_completion_tokens: int
    music_theory_final_advice_prompt_tokens: int
    music_theory_final_advice_completion_tokens: int
    instruments_final_advice_prompt_tokens: int
    instruments_final_advice_completion_tokens: int
    lyrics_final_advice_prompt_tokens: int
    lyrics_final_advice_completion_tokens: int
    production_final_advice_prompt_tokens: int
    production_final_advice_completion_tokens: int
    
    # Total accumulated token counters
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: Optional[float]
    
    # --- Final Output ---
    final_moodboard: Optional[str]      # The combined Markdown output

    # --- Control Flags & Error Handling (Optional but useful) ---
    # You could add flags to conditionally run certain agents
    # For example, if the user query doesn't imply lyrics, this could be set to False.
    should_run_lyrics_agent: bool
    error_message: Optional[str]        # To store any error messages encountered during the flow

   