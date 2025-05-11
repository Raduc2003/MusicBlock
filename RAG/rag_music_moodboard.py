# RAG/run_music_moodboard.py

import sys
from typing import Dict, Any, Optional

# Ensure the RAG directory is in the Python path for relative imports if running directly
# This is often needed if you run 'python RAG/run_music_moodboard.py' from the project root
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#    sys.path.append(current_dir)
# if os.path.dirname(current_dir) not in sys.path: # If RAG is a subdir of project root
#    sys.path.append(os.path.dirname(current_dir))


from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # Optional: for state persistence

# Import state, nodes, and configs
from rag_state import OverallState
from rag_agent_nodes import (
    process_initial_input_node,
    rhythm_agent_node,
    music_theory_agent_node,
    instruments_agent_node,
    lyrics_agent_node,
    production_agent_node,
    combine_advice_node
)
from rag_config import AGENT_TOPICS # To know which agents to wire up

# --- Graph Definition ---

# Helper function to decide if lyrics agent should run
def should_run_lyrics(state: OverallState) -> str:
    """Determines the next node after initial processing based on lyrics flag."""
    if state.get("error_message"): # If initial processing had an error
        print("ROUTER: Error in initial processing, proceeding to combiner to output error.")
        return "combine_advice_node" # Go directly to output if there was an error

    if state.get("should_run_lyrics_agent", False):
        print("ROUTER: Lyrics agent will run.")
        return "lyrics_agent" # Name of the lyrics agent node
    else:
        print("ROUTER: Lyrics agent will be skipped.")
        # If lyrics agent is skipped, other agents still run in parallel,
        # then all proceed to combiner. We need a way to gather parallel branches.
        # LangGraph handles this if edges from parallel nodes point to the same next node.
        # For this simple conditional branch, we can route to a common next step or directly to combiner
        # if it's the only conditional one not in the main parallel flow.
        # Let's assume for now it's part of the parallel flow and simply doesn't populate its advice.
        # A more explicit skip would be to have an edge from the router to the combiner if lyrics are skipped
        # and lyrics_agent is not added to the list of parallel nodes.
        # For now, lyrics_agent_node itself handles not doing work.
        # The router's job here is more about *which* path to take IF there were exclusive paths.
        # In a parallel setup, this router might be less about skipping a parallel node and more about
        # deciding on a subsequent step *after* the parallel block, if lyrics were a gate.
        # Let's simplify: process_initial_input will set the flag, lyrics_agent will respect it.
        # The main parallel paths will include lyrics if the flag is true.
        # For now, we'll make the main path unconditional to all agents for simplicity in edges,
        # and the lyrics_agent_node itself will do nothing if the flag is false.
        # So, this router isn't strictly needed if lyrics_agent_node handles the skip.
        # However, if we wanted to truly skip adding it to a parallel execution list,
        # the routing would be more complex after 'process_initial_input_node'.

        # Let's define parallel branches first, then consider conditional routing.
        # For now, all specialist agents are called after initial_processor.
        # The 'should_run_lyrics_agent' flag in the state is used by the lyrics_agent_node itself.
        # So, no complex routing logic needed here for just that.
        # This function might be used if we had mutually exclusive paths.
        return "trigger_specialist_agents" # A dummy node or directly to the first specialist if sequential


# Create the StateGraph
workflow = StateGraph(OverallState)

# 1. Add the nodes
workflow.add_node("initial_processor", process_initial_input_node)

# Specialist agent nodes
workflow.add_node("rhythm_agent", rhythm_agent_node)
workflow.add_node("music_theory_agent", music_theory_agent_node)
workflow.add_node("instruments_agent", instruments_agent_node)
workflow.add_node("lyrics_agent", lyrics_agent_node) # Will check its flag internally
workflow.add_node("production_agent", production_agent_node)

workflow.add_node("advice_combiner", combine_advice_node)

# 2. Define the edges (the flow)
workflow.set_entry_point("initial_processor")

# After initial processing, trigger all specialist agents.
# LangGraph can run subsequent nodes in parallel if they don't have direct sequential dependencies.
# To make them run "in parallel" conceptually from initial_processor's output,
# we add edges from initial_processor to each of them.
# They will all receive the state from initial_processor.
workflow.add_edge("initial_processor", "rhythm_agent")
workflow.add_edge("initial_processor", "music_theory_agent")
workflow.add_edge("initial_processor", "instruments_agent")
workflow.add_edge("initial_processor", "lyrics_agent") # lyrics_agent checks its own flag
workflow.add_edge("initial_processor", "production_agent")


# After all specialist agents have run, their outputs will be in the state.
# Then, proceed to the advice_combiner.
# For this to work correctly with parallel execution, LangGraph needs to know
# when all parallel branches are complete before moving to the combiner.
# We need to ensure all these agent nodes eventually lead to the 'advice_combiner'.
# If they run in parallel, they don't directly connect to each other but to the next join point.

# To make them join at the combiner:
workflow.add_edge("rhythm_agent", "advice_combiner")
workflow.add_edge("music_theory_agent", "advice_combiner")
workflow.add_edge("instruments_agent", "advice_combiner")
workflow.add_edge("lyrics_agent", "advice_combiner")
workflow.add_edge("production_agent", "advice_combiner")


workflow.add_edge("advice_combiner", END)


# Optional: For state persistence and resuming (more advanced)
# memory = SqliteSaver.from_conn_string(":memory:") # In-memory checkpointing
# app = workflow.compile(checkpointer=memory)
app = workflow.compile()

print("LangGraph RAG Moodboard Generator Compiled.")

# RAG/run_music_moodboard.py

# ... (other imports and graph definition remain the same) ...

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Running Music Moodboard RAG ---")

    # --- 1. Get User Input ---
    user_text_query = input("Enter your music idea (e.g., 'dark pop song like The Weeknd with strings'): ")
    
    user_audio_path_input = input("Enter path to your audio file (or press Enter to skip): ").strip()
    if not user_audio_path_input:
        user_audio_path = None
        print("No audio file provided.")
    else:
        user_audio_path = user_audio_path_input
        print(f"Audio file provided: {user_audio_path}")


    # --- 2. Placeholder for Music Similarity Pipeline ---
    similar_tracks_summary_output: Optional[str]

    # This is your example table data
    similar_tracks_table_data = """
| #  | Title                | Artist                          | Key       | BPM | Genres                      | Other tags                          |
| -- | -------------------- | ------------------------------- | --------- | --- | --------------------------- | ----------------------------------- |
| 1  | Serge’s Kiss         | Daybehavior                     | C minor   | 109 | alternative rock, dream pop | re-recording, pop, alternative rock |
| 2  | Imagine              | John Lennon                     | C major   | 75  | rock, pop                   | piano, classic, 1971                |
| 3  | Billie Jean          | Michael Jackson                 | F ♯ minor | 117 | pop, R&B                    | dance, 1980s, synth                 |
| 4  | Smells Like Teen…    | Nirvana                         | F minor   | 117 | grunge, alternative rock    | 1990s, guitar riff, breakthrough    |
| 5  | Rolling in the Deep  | Adele                           | C minor   | 105 | pop, soul                   | powerful vocals, 2010s              |
| 6  | Take Five            | The Dave Brubeck Quartet        | E ♭ minor | 174 | jazz, cool jazz             | saxophone, classic, instrumental    |
| 7  | Get Lucky            | Daft Punk ft. Pharrell Williams | F minor   | 116 | disco, electronic, funk     | dancefloor, 2010s                   |
| 8  | Nothing Else Matters | Metallica                       | E minor   | 142 | heavy metal, rock           | ballad, acoustic intro              |
| 9  | Bad Guy              | Billie Eilish                   | G minor   | 135 | pop, electro-pop            | whisper vocals, modern sound        |
| 10 | Clocks               | Coldplay                        | E ♭ major | 131 | alternative rock, pop rock  | piano riff, 2000s                   |
"""

    if user_audio_path:
        print(f"\nSIMULATING Music Similarity Pipeline for: {user_audio_path}...")
        # Use the provided table as the summary output when audio is present
        similar_tracks_summary_output = f"Analysis of acoustically similar tracks to the input audio yielded the following results:\n{similar_tracks_table_data}"
        print(f"Similarity Summary (Placeholder):\n{similar_tracks_summary_output}\n")
    else:
        # If no audio, we can choose to provide a generic message or try to infer from text only
        # For now, let's still provide the table as a general "example tracks" if no audio,
        # or you could have a different placeholder.
        # Alternatively, just state no audio analysis was done.
        # Let's make it distinct if no audio.
        similar_tracks_summary_output = (
            "No audio provided for similarity analysis. "
            "However, for context, here are some diverse tracks that might provide general inspiration:\n"
            f"{similar_tracks_table_data}"
            "\n(Note: These are general examples, not based on specific audio input.)"
        )
        print(f"{similar_tracks_summary_output}\n")


    # --- 3. Prepare Initial State for the Graph ---
    initial_graph_state = {
        "user_text_query": user_text_query,
        "user_audio_path": user_audio_path,
        "similar_tracks_summary": similar_tracks_summary_output, # Output from your separate pipeline (or placeholder)
        "project_goal_summary": "", 
        "rhythm_advice": None,
        "music_theory_advice": None,
        "instruments_advice": None,
        "lyrics_advice": None,
        "production_advice": None,
        "final_moodboard": None,
        "error_message": None,
        "should_run_lyrics_agent": False 
    }

    # --- 4. Invoke the LangGraph Workflow ---
    # ... (rest of the script remains the same: invoke app, print output) ...
    print("Invoking LangGraph workflow...")
    try:
        final_result_state = app.invoke(initial_graph_state)

        print("\n\n--- Generated Music Inspiration Moodboard ---")
        if final_result_state and final_result_state.get("final_moodboard"):
            print(final_result_state["final_moodboard"])
        elif final_result_state and final_result_state.get("error_message"):
            print(f"An error occurred: {final_result_state['error_message']}")
        else:
            print("No moodboard generated or an unknown error occurred.")
            print("Final state dump:", final_result_state)

    except Exception as e:
        print(f"\nAn unexpected error occurred while running the graph: {e}")
        import traceback
        traceback.print_exc()