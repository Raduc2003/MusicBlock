# RAG/rag_agent_nodes.py

from typing import Dict, List, Optional, Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# Assuming these files are in the same RAG directory, hence relative imports
from rag_state import OverallState
from rag_tools import get_knowledge_from_kb
from rag_config import (
    LLM_PROVIDER,
    LOCAL_LLM_API_BASE,
    LOCAL_LLM_MODEL_NAME,
    LOCAL_LLM_API_KEY,
    DEFAULT_RETRIEVAL_TOP_K
)
from rag_prompts import (
    INITIAL_GOAL_SYNTHESIS_PROMPT_TEMPLATE,
    RHYTHM_ADVICE_PROMPT_TEMPLATE,
    MUSIC_THEORY_ADVICE_PROMPT_TEMPLATE,
    INSTRUMENTS_ADVICE_PROMPT_TEMPLATE,
    LYRICS_ADVICE_PROMPT_TEMPLATE,
    PRODUCTION_ADVICE_PROMPT_TEMPLATE
)

# --- LLM Initialization ---
llm = None
try:
    if LLM_PROVIDER == "LocalOpenAICompatible":
        api_key_to_use = None
        if LOCAL_LLM_API_KEY and LOCAL_LLM_API_KEY.lower() not in ["notneeded", "none", ""]:
            api_key_to_use = LOCAL_LLM_API_KEY
        elif LOCAL_LLM_API_KEY and LOCAL_LLM_API_KEY.lower() in ["notneeded", "none", ""]:
             api_key_to_use = "NotNeeded" # Langchain OpenAI client seems to want a string

        llm = ChatOpenAI(
            model=LOCAL_LLM_MODEL_NAME,
            openai_api_base=LOCAL_LLM_API_BASE,
            api_key=api_key_to_use,
            temperature=0.7,
            # default_headers={"Content-Type": "application/json"}, # Already handled by client
        )
        print(f"LLM Initialized for rag_agent_nodes: LocalOpenAICompatible with model {LOCAL_LLM_MODEL_NAME} at {LOCAL_LLM_API_BASE} (API Key Used: '{api_key_to_use}')")
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER} in rag_agent_nodes.")
except Exception as e:
    print(f"FATAL: Could not initialize LLM in rag_agent_nodes: {e}")
    # llm remains None


# --- Node 1: Input Processing & Goal Synthesis ---
def process_initial_input_node(state: OverallState) -> Dict[str, Any]:
    print("--- NODE: Processing Initial Input & Synthesizing Goal ---")
    node_error_message = None
    project_goal_summary = "Project goal could not be determined." # Default

    if not llm:
        project_goal_summary = "CRITICAL ERROR: LLM not initialized for project goal synthesis."
        node_error_message = "LLM not initialized for the entire graph. Cannot proceed."
    else:
        user_query = state["user_text_query"]
        similar_tracks_summary = state.get("similar_tracks_summary", "No specific similarity analysis provided.")
        prompt_input = {"user_text_query": user_query, "similar_tracks_summary": similar_tracks_summary}
        prompt = INITIAL_GOAL_SYNTHESIS_PROMPT_TEMPLATE.format(**prompt_input)
        
        print(f"NODE (process_initial_input): Generating project goal summary...")
        try:
            response = llm.invoke(prompt)
            project_goal_summary = response.content
            print(f"NODE (process_initial_input): Project Goal Summary successfully generated.")
        except Exception as e:
            print(f"CRITICAL ERROR during project goal synthesis LLM call: {e}")
            project_goal_summary = f"Error: Could not synthesize project goal. Details: {e}"
            node_error_message = f"Failed to synthesize project goal: {e}"

    should_run_lyrics = "lyric" in state["user_text_query"].lower() or "vocal" in state["user_text_query"].lower()
    
    output_dict: Dict[str, Any] = {
        "project_goal_summary": project_goal_summary,
        "should_run_lyrics_agent": should_run_lyrics
    }
    if node_error_message: # If this critical first step fails, set the global error
        output_dict["error_message"] = node_error_message
    return output_dict


# --- Specialist Agent Node Generic Logic ---
def _specialist_agent_node_logic(
    agent_name: str,
    state: OverallState,
    knowledge_topic: str, # CRITICAL: This must match Qdrant metadata.topic
    prompt_template: str,
    retrieved_chunks_key_in_prompt: str
) -> Dict[str, str]:
    advice_key = f"{agent_name.lower().replace(' ', '_')}_advice"
    advice_content = f"{agent_name} advice could not be generated." # Default

    print(f"--- NODE: {agent_name} Agent ---")

    if state.get("error_message"): # Check for critical error from previous steps
        advice_content = f"Skipping {agent_name} advice due to earlier critical error: {state.get('error_message')}"
        print(f"NODE ({agent_name}_agent): {advice_content}")
        return {advice_key: advice_content}
    if not llm:
        advice_content = f"Error: LLM not initialized for {agent_name} agent."
        print(f"NODE ({agent_name}_agent): {advice_content}")
        return {advice_key: advice_content}
    
    project_goal = state.get("project_goal_summary")
    if not project_goal or "Error:" in project_goal.split('.')[0]: # Check if goal itself indicates an error
        advice_content = f"Skipping {agent_name} advice: Project goal not available or in error state ('{project_goal[:100]}...')."
        print(f"NODE ({agent_name}_agent): {advice_content}")
        return {advice_key: advice_content}

    try:
        search_query = f"{agent_name} concepts and techniques for: {project_goal}"
        print(f"NODE ({agent_name}_agent): Retrieving knowledge for topic '{knowledge_topic}' with query: '{search_query[:100]}...'")
        
        retrieved_docs = get_knowledge_from_kb(
            search_query=search_query,
            topic=knowledge_topic, # Use the specific topic for this agent
            top_k=DEFAULT_RETRIEVAL_TOP_K
        )
        context_chunks_text = "\n\n---\n\n".join([
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        ]) if retrieved_docs else f"No specific knowledge for '{knowledge_topic}' retrieved from the knowledge base."
        
        prompt_input = {
            "project_goal_summary": project_goal,
            retrieved_chunks_key_in_prompt: context_chunks_text
        }
        prompt = prompt_template.format(**prompt_input)
        
        print(f"NODE ({agent_name}_agent): Generating {agent_name.lower()} advice...")
        response = llm.invoke(prompt)
        advice_content = response.content
        print(f"NODE ({agent_name}_agent): {agent_name} Advice successfully generated.")
    except Exception as e:
        print(f"Error during {agent_name} agent's own execution (LLM call or retrieval): {e}")
        advice_content = f"Error: Could not generate {agent_name.lower()} advice. Details: {e}"
        
    return {advice_key: advice_content}


# --- Specialist Agent Node Definitions ---
def rhythm_agent_node(state: OverallState) -> Dict[str, str]:
    return _specialist_agent_node_logic(
        agent_name="Rhythm",
        state=state,
        knowledge_topic="rythm", # VERIFY THIS TOPIC NAME with your Qdrant data
        prompt_template=RHYTHM_ADVICE_PROMPT_TEMPLATE,
        retrieved_chunks_key_in_prompt="retrieved_rhythm_chunks"
    )

def music_theory_agent_node(state: OverallState) -> Dict[str, str]:
    return _specialist_agent_node_logic(
        agent_name="Music Theory",
        state=state,
        knowledge_topic="theory_general", # VERIFY THIS TOPIC NAME
        prompt_template=MUSIC_THEORY_ADVICE_PROMPT_TEMPLATE,
        retrieved_chunks_key_in_prompt="retrieved_music_theory_chunks"
    )

def instruments_agent_node(state: OverallState) -> Dict[str, str]:
    return _specialist_agent_node_logic(
        agent_name="Instruments",
        state=state,
        knowledge_topic="timbre_instruments", # VERIFY THIS TOPIC NAME
        prompt_template=INSTRUMENTS_ADVICE_PROMPT_TEMPLATE,
        retrieved_chunks_key_in_prompt="retrieved_instruments_chunks"
    )

def lyrics_agent_node(state: OverallState) -> Dict[str, str]:
    print("--- NODE: Lyrics Agent ---")
    advice_content = "" # Default to empty if skipped or other pre-checks fail

    if not state.get("should_run_lyrics_agent", False):
        print("NODE (Lyrics_agent): Skipping as per should_run_lyrics_agent flag.")
        return {"lyrics_advice": advice_content}

    # Now call the generic logic if it's supposed to run
    return _specialist_agent_node_logic(
        agent_name="Lyrics",
        state=state,
        knowledge_topic=None, # VERIFY THIS TOPIC NAME (or set to None for general search)
        prompt_template=LYRICS_ADVICE_PROMPT_TEMPLATE,
        retrieved_chunks_key_in_prompt="retrieved_lyrics_chunks"
    )

def production_agent_node(state: OverallState) -> Dict[str, str]:
    return _specialist_agent_node_logic(
        agent_name="Production",
        state=state,
        knowledge_topic="production", # VERIFY THIS TOPIC NAME
        prompt_template=PRODUCTION_ADVICE_PROMPT_TEMPLATE,
        retrieved_chunks_key_in_prompt="retrieved_production_chunks"
    )


# --- Node for Combining Advice ---
def combine_advice_node(state: OverallState) -> Dict[str, Any]:
    print("--- NODE: Combining Advice ---")
    project_goal = state.get("project_goal_summary", "N/A (Project goal was not generated or was missing)")
    initial_critical_error = state.get("error_message")

    moodboard_parts = []

    if initial_critical_error:
        moodboard_parts.append(f"# Music Inspiration Moodboard - Generation Failed\n\n## Critical Error During Setup\n{initial_critical_error}\n")
        moodboard_parts.append(f"Attempted Project Goal:\n{project_goal}\n")
    else:
        moodboard_parts.append(f"# Music Inspiration Moodboard\n\n## Project Goal\n{project_goal}\n")

    # Helper to add advice sections, including embedded errors
    def add_advice_section(title: str, advice_key: str, is_conditional: bool = False, condition_met: bool = True):
        if is_conditional and not condition_met:
            # Only add "skipped" message if it was meant to run but didn't produce content for other reasons than skip flag
            # The lyrics_agent_node now returns "" if skipped by flag, so this handles it.
            # If advice is None, it means the key wasn't even returned by the agent.
            # if state.get(advice_key) is None: # If key is missing entirely (shouldn't happen if agent ran)
            #    moodboard_parts.append(f"## {title}\n(This section was not processed)")
            return # Do not add section if condition to run was not met AND it correctly returned empty/None

        advice_content = state.get(advice_key)
        if advice_content: # Non-empty string means advice or an embedded error
            moodboard_parts.append(f"## {title}\n{advice_content}")
        elif advice_content == "": # Specifically for agents that can be skipped and return empty string
             moodboard_parts.append(f"## {title}\n(This section was intentionally skipped or produced no content)")
        # If None, and it was supposed to run, something unexpected happened.
        elif advice_content is None and (not is_conditional or condition_met):
            moodboard_parts.append(f"## {title}\n(No advice was generated or an error occurred in this section - state key missing)")


    add_advice_section("Rhythm & Groove", "rhythm_advice")
    add_advice_section("Music Theory & Harmony", "music_theory_advice")
    add_advice_section("Timbre & Instrumentation", "instruments_advice")
    add_advice_section("Lyrics & Vocals", "lyrics_advice", is_conditional=True, condition_met=state.get("should_run_lyrics_agent", False))
    add_advice_section("Production & Mix", "production_advice")
    
    final_moodboard = "\n\n".join(moodboard_parts)
    print("NODE (combine_advice): Final moodboard assembled.")
    
    # Return the final moodboard and preserve the critical error message if one existed
    output_state: Dict[str, Any] = {"final_moodboard": final_moodboard}
    if initial_critical_error:
        output_state["error_message"] = initial_critical_error # Propagate if set
    return output_state