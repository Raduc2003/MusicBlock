# RAG/run_music_moodboard.py

import sys
import os
import datetime
import logging # Using Python's standard logging
import argparse
from typing import Dict, Any, Optional
import graphviz

from langgraph.graph import StateGraph, END
from rag_state import OverallState
from rag_agent_nodes import (
    process_initial_input_node,
    rhythm_agent_node,
    music_theory_agent_node,
    instruments_agent_node,
    lyrics_agent_node,
    production_agent_node,
    combine_advice_node,
    llm as agent_llm 
)
from rag_config import AGENT_TOPICS, LLM_MODEL_NAME 

# --- Logging Setup ---
def setup_logging(log_level_console_str="INFO", log_level_file_str="DEBUG"):
    log_dir = "rag_logs"
    log_filename = "moodboard_run_error.log"
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[SETUP_LOGGING_PRINT] CREATED Log directory: {os.path.abspath(log_dir)}")
    except Exception as e:
        print(f"[SETUP_LOGGING_PRINT] ERROR creating log directory {log_dir}: {e}")
        
        logging.basicConfig(level=getattr(logging, log_level_console_str.upper(), logging.INFO),
                            format='%(levelname)-8s: %(name)-20s: %(message)s')
        return logging.getLogger(__name__)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"moodboard_run_{timestamp}.log")
  
    print(f"[SETUP_LOGGING_PRINT] INTENDING TO LOG TO FILE: {os.path.abspath(log_filename)}")

    file_level = getattr(logging, log_level_file_str.upper(), logging.DEBUG)
    console_level = getattr(logging, log_level_console_str.upper(), logging.INFO)

    root_logger = logging.getLogger()
   
    root_logger.setLevel(logging.DEBUG) 
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    try:
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(file_level) # e.g., DEBUG
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(name)-30s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
    except Exception as e:
        print(f"[SETUP_LOGGING_PRINT] ERROR setting up file handler for {log_filename}: {e}")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level) # e.g., INFO
    console_formatter = logging.Formatter('%(levelname)-8s: %(name)-25s: %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    # print(f"[SETUP_LOGGING_PRINT] Console handler added at level {logging.getLevelName(console_level)}")
    
    module_logger = logging.getLogger(__name__) # Logger for this current file (__main__)
    module_logger.info(f"Logging initialized. Console: {logging.getLevelName(console_level)}, File: {logging.getLevelName(file_level)} -> {os.path.abspath(log_filename)}")
    return module_logger
logger = setup_logging(log_level_file_str="DEBUG", log_level_console_str="INFO")
logger.info("RAG Music Moodboard Script Started (after local module imports).")
logger.info(f"Using LLM: {LLM_MODEL_NAME} (Status from agent_nodes: {'INITIALIZED' if agent_llm else 'NOT INITIALIZED IN AGENT NODES'})")




# --- Graph Definition ---
def build_graph(sequential_mode: bool = False) -> StateGraph:
    workflow = StateGraph(OverallState)
    logger.info("Building graph: Adding nodes...")
    workflow.add_node("initial_processor", process_initial_input_node)
    workflow.add_node("rhythm_agent", rhythm_agent_node)
    workflow.add_node("music_theory_agent", music_theory_agent_node)
    workflow.add_node("instruments_agent", instruments_agent_node)
    workflow.add_node("lyrics_agent", lyrics_agent_node)
    workflow.add_node("production_agent", production_agent_node)
    workflow.add_node("advice_combiner", combine_advice_node)

    workflow.set_entry_point("initial_processor")
    logger.info("Building graph: Entry point set to 'initial_processor'.")

    if sequential_mode:
        logger.info("Building graph: Using SEQUENTIAL agent execution flow.")
        workflow.add_edge("initial_processor", "rhythm_agent")
        workflow.add_edge("rhythm_agent", "music_theory_agent")
        workflow.add_edge("music_theory_agent", "instruments_agent")
        workflow.add_edge("instruments_agent", "lyrics_agent")
        workflow.add_edge("lyrics_agent", "production_agent")
        workflow.add_edge("production_agent", "advice_combiner")
    else: 
        logger.info("Building graph: Using CONCURRENT agent execution flow.")
        workflow.add_edge("initial_processor", "rhythm_agent")
        workflow.add_edge("initial_processor", "music_theory_agent")
        workflow.add_edge("initial_processor", "instruments_agent")
        workflow.add_edge("initial_processor", "lyrics_agent")
        workflow.add_edge("initial_processor", "production_agent")


        workflow.add_edge("rhythm_agent", "advice_combiner")
        workflow.add_edge("music_theory_agent", "advice_combiner")
        workflow.add_edge("instruments_agent", "advice_combiner")
        workflow.add_edge("lyrics_agent", "advice_combiner")
        workflow.add_edge("production_agent", "advice_combiner")

    workflow.add_edge("advice_combiner", END)
    logger.info("Building graph: Final edge to END set from 'advice_combiner'.")
    return workflow




# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Inspiration Moodboard RAG Generator")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run specialist agent nodes sequentially instead of concurrently."
    )
    parser.add_argument(
        "--debugconsole",
        action="store_true",
        help="Enable DEBUG level logging for the console output."
    )
    args = parser.parse_args()

 
    if args.debugconsole:
        logger.info("Setting CONSOLE log level to DEBUG due to --debugconsole flag.")
        for handler in logging.getLogger().handlers: 
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(logging.DEBUG)
                logger.info("Console log level set to DEBUG.")
                break 

    logger.info(f"--- New Moodboard Generation Run ---")
    logger.info(f"Execution Mode: {'SEQUENTIAL' if args.sequential else 'CONCURRENT'}")


    workflow_graph = build_graph(sequential_mode=args.sequential)


    AGENTS = [
    "rhythm_agent",
    "music_theory_agent",
    "instruments_agent",
    "lyrics_agent",
    "production_agent",
]

    def visualize_state_graph(sequential: bool, filename="moodboard_graph"):
        """
        Render the nodes & edges exactly as in build_graph() into
        filename.png (and .dot).
        """
        dot = graphviz.Digraph(comment="Music Moodboard RAG")
        # all the nodes
        dot.node("initial_processor")
        for a in AGENTS:
            dot.node(a)
        dot.node("advice_combiner")
        dot.node("END")  # END is a string, not an object

        # edges
        if sequential:
            seq = ["initial_processor"] + AGENTS + ["advice_combiner"]
            for src, dst in zip(seq, seq[1:]):
                dot.edge(src, dst)
        else:
            # fan‐out from initial
            for a in AGENTS:
                dot.edge("initial_processor", a)
            # fan‐in to combiner
            for a in AGENTS:
                dot.edge(a, "advice_combiner")
        # final edge to END
        dot.edge("advice_combiner", "END")
        
        try:
            # Render the graph
            dot.render(filename, format='png', cleanup=True)
            logger.info(f"Graph visualization saved as {filename}.png")
        except Exception as e:
            logger.warning(f"Graphviz render failed: {e}")

    # Call the visualization function
    try:
        visualize_state_graph(args.sequential)
    except Exception as e:
        logger.warning(f"Graphviz visualization failed: {e}")

    app = workflow_graph.compile()
    logger.info("LangGraph RAG Moodboard Generator Compiled.")

    user_text_query = input("Enter your music idea (e.g., 'dark pop song like The Weeknd with strings'): ")
    logger.info(f"User Text Query: {user_text_query}")
    
    user_audio_path_input = input("Enter path to your audio file (or press Enter to skip): ").strip()
    user_audio_path: Optional[str] = None # Type hint for clarity
    if not user_audio_path_input:
        logger.info("No audio file provided by user.")
    else:
        user_audio_path = user_audio_path_input
        logger.info(f"User Audio File Path: {user_audio_path}")

    similar_tracks_summary_output: Optional[str] # Type hint
    similar_tracks_table_data = """
| #  | Title                | Artist                          | Key       | BPM | Genres                      | Other tags                          |
| -- | -------------------- | ------------------------------- | --------- | --- | --------------------------- | ----------------------------------- |
| 1  | Serge's Kiss         | Daybehavior                     | C minor   | 109 | alternative rock, dream pop | re-recording, pop, alternative rock |
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
        logger.info(f"SIMULATING Music Similarity Pipeline for: {user_audio_path}...")
        similar_tracks_summary_output = f"Analysis of acoustically similar tracks to the input audio yielded the following results:\n{similar_tracks_table_data}"
        logger.info(f"Similarity Summary (Placeholder) Snippet:\n{similar_tracks_summary_output[:200]}...")
    else:
        similar_tracks_summary_output = f"Analysis of acoustically similar tracks to the input audio yielded the following results:\n{similar_tracks_table_data}"
        logger.info(f"Similarity Summary (Placeholder) Snippet:\n{similar_tracks_summary_output[:200]}...")

    initial_graph_state = OverallState(
        user_text_query=user_text_query,
        user_audio_path=user_audio_path,
        user_audio_features=None,
        similar_tracks_summary=similar_tracks_summary_output,
        project_goal_summary="", 
        rhythm_advice=None, rhythm_kb_sources=[], rhythm_stack_sources=[],
        music_theory_advice=None, music_theory_kb_sources=[], music_theory_stack_sources=[],
        instruments_advice=None, instruments_kb_sources=[], instruments_stack_sources=[],
        lyrics_advice=None, lyrics_kb_sources=[], lyrics_stack_sources=[], # Ensure these are initialized
        production_advice=None, production_kb_sources=[], production_stack_sources=[],
        final_moodboard=None,
        error_message=None,
        should_run_lyrics_agent=False,
        all_accumulated_sources=[],
        node_prompt_tokens=0,
        node_completion_tokens=0,
        # Agent-specific token counters for SE query generation
        rhythm_se_query_prompt_tokens=0,
        rhythm_se_query_completion_tokens=0,
        music_theory_se_query_prompt_tokens=0,
        music_theory_se_query_completion_tokens=0,
        instruments_se_query_prompt_tokens=0,
        instruments_se_query_completion_tokens=0,
        lyrics_se_query_prompt_tokens=0,
        lyrics_se_query_completion_tokens=0,
        production_se_query_prompt_tokens=0,
        production_se_query_completion_tokens=0,
        # Agent-specific token counters for final advice generation
        rhythm_final_advice_prompt_tokens=0,
        rhythm_final_advice_completion_tokens=0,
        music_theory_final_advice_prompt_tokens=0,
        music_theory_final_advice_completion_tokens=0,
        instruments_final_advice_prompt_tokens=0,
        instruments_final_advice_completion_tokens=0,
        lyrics_final_advice_prompt_tokens=0,
        lyrics_final_advice_completion_tokens=0,
        production_final_advice_prompt_tokens=0,
        production_final_advice_completion_tokens=0,
        # Total accumulated token counters
        total_prompt_tokens=0, # Initialize totals
        total_completion_tokens=0,
        total_cost=0.0,
    )
    logger.info(f"Initial graph state prepared (keys only): {list(initial_graph_state.keys())}")

    logger.info("Invoking LangGraph workflow...")
    logger.info("=== WORKFLOW EXECUTION START ===")
    final_result_state: Optional[OverallState] = None
    try:
        logger.info("Calling app.invoke() with initial state...")
        final_result_state = app.invoke(initial_graph_state) 
        logger.info("=== WORKFLOW EXECUTION COMPLETE ===")
        logger.info("Graph invocation complete.")
        if final_result_state: 
            logger.debug(f"Final State Dump: {final_result_state}")
            logger.info("=== TOKEN CALCULATION START ===")
            calculated_total_prompt = 0
            calculated_total_completion = 0
            calculated_total_prompt += final_result_state.get("node_prompt_tokens", 0) 
            calculated_total_completion += final_result_state.get("node_completion_tokens", 0)

            agent_prefixes = ["rhythm", "music_theory", "instruments", "lyrics", "production"]
            logger.info("Summing agent-specific token counts...")
            for prefix in agent_prefixes:
                se_q_p = final_result_state.get(f"{prefix}_se_query_prompt_tokens", 0)
                se_q_c = final_result_state.get(f"{prefix}_se_query_completion_tokens", 0)
                final_p = final_result_state.get(f"{prefix}_final_advice_prompt_tokens", 0)
                final_c = final_result_state.get(f"{prefix}_final_advice_completion_tokens", 0)
                
                calculated_total_prompt += se_q_p + final_p
                calculated_total_completion += se_q_c + final_c
                
                if se_q_p or se_q_c or final_p or final_c:
                    logger.info(f"Agent {prefix}: SE({se_q_p}+{se_q_c}) + Final({final_p}+{final_c}) = {se_q_p + se_q_c + final_p + final_c} total")
            
           

            logger.debug("All token-related keys in final state:")
            for key, value in final_result_state.items():
                if "tokens" in key.lower() and isinstance(value, int) and value > 0:
                    logger.debug(f"Token usage from state key '{key}': {value}")
            
            logger.info(f"Calculated Total Prompt Tokens (summed post-run): {calculated_total_prompt}")
            logger.info(f"Calculated Total Completion Tokens (summed post-run): {calculated_total_completion}")
            logger.info(f"Overall Total Tokens: {calculated_total_prompt + calculated_total_completion}")

            tp = final_result_state.get("total_prompt_tokens", 0)
            tc = final_result_state.get("total_completion_tokens", 0)
            logger.info(f"👉 Total prompt tokens (from combiner):     {tp}")
            logger.info(f"👉 Total completion tokens (from combiner): {tc}")
            logger.info(f"👉 Grand total tokens (from combiner):      {tp + tc}")
            logger.info("=== TOKEN CALCULATION COMPLETE ===")

            
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the graph: {e}", exc_info=True)
     
    logger.info("--- Moodboard Generation Attempt Finished ---")
    if final_result_state:
        final_moodboard_content = final_result_state.get("final_moodboard")
        if final_moodboard_content:
            logger.info("🎉 MOODBOARD GENERATION SUCCESSFUL!")
            logger.info("=== FINAL GENERATED MOODBOARD CONTENT ===")
            logger.info(final_moodboard_content)
            logger.info("=== END FINAL MOODBOARD CONTENT ===")
            logger.info(f"Final moodboard length: {len(final_moodboard_content)} characters")
            
            print("\n\n" + "="*30 + " Generated Music Inspiration Moodboard " + "="*30)
            print(final_moodboard_content)
            print("="*80 + "\n")
            logger.info("Successfully displayed final moodboard to user.")
        
        error_msg_content = final_result_state.get("error_message")
        if error_msg_content:
            logger.error(f"Moodboard generation failed with critical error: {error_msg_content}")
            print(f"\nMOODBOARD GENERATION FAILED: {error_msg_content}")
        elif not final_moodboard_content and not error_msg_content:
            logger.warning("No final moodboard content, but no critical error. Check detailed logs.")
            print("\nNo moodboard content. Check logs.")
    else:
        logger.error("Graph execution did not produce a final state.")
        print("\nGraph execution did not produce a final state. Check logs for errors.")

    file_log_path = "Log file path not found (FileHandler not identified)."
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            file_log_path = handler.baseFilename
            break
    logger.info(f"Log file for this run: {file_log_path}")
