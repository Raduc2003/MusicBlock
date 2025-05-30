# RAG/rag_agent_nodes.py

import logging
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import OpenAICallbackHandler

from rag_state import OverallState
from rag_tools import get_knowledge_from_kb, search_stackexchange_qa
from rag_config import (
    LLM_API_BASE,
    LLM_MODEL_NAME,
    LLM_API_KEY,
    DEFAULT_RETRIEVAL_TOP_K,
)
from rag_prompts import (
    INITIAL_GOAL_SYNTHESIS_PROMPT_TEMPLATE,
    STACKEXCHANGE_QUESTION_GENERATION_PROMPT_TEMPLATE,
    RHYTHM_ADVICE_PROMPT_TEMPLATE,
    MUSIC_THEORY_ADVICE_PROMPT_TEMPLATE,
    INSTRUMENTS_ADVICE_PROMPT_TEMPLATE,
    LYRICS_ADVICE_PROMPT_TEMPLATE,
    PRODUCTION_ADVICE_PROMPT_TEMPLATE,
    RHYTHM_ADVICE_COT_PROMPT_TEMPLATE,
    MUSIC_THEORY_ADVICE_COT_PROMPT_TEMPLATE, 
    INSTRUMENTS_ADVICE_COT_PROMPT_TEMPLATE,  
    LYRICS_ADVICE_COT_PROMPT_TEMPLATE,       
    PRODUCTION_ADVICE_COT_PROMPT_TEMPLATE 
)

node_logger = logging.getLogger(__name__)
if not node_logger.handlers and not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

# --- Initialize LLM client ---
llm = None
try:
    if not all([LLM_API_BASE, LLM_MODEL_NAME]):
        raise ValueError("LLM_API_BASE or LLM_MODEL_NAME not configured.")
    api_key = (
        LLM_API_KEY
        if LLM_API_KEY and LLM_API_KEY.lower() not in ["notneeded", "none", ""]
        else None
    )
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        base_url=LLM_API_BASE,
        api_key=api_key,
        temperature=0.7,
        request_timeout=200,  
    )
    node_logger.info(f"LLM initialized: Model={LLM_MODEL_NAME}, Base={LLM_API_BASE}")
except Exception as e:
    node_logger.error(f"FATAL: LLM init failed: {e}", exc_info=True)


def process_initial_input_node(state: OverallState) -> Dict[str, Any]:
    node_logger.info("--- NODE START: Initial Input Processor ---")
    error_msg = None
    goal_summary = "Project goal determination failed."
    prompt_tokens = 0
    completion_tokens = 0

    if not llm:
        error_msg = "LLM not initialized for graph."
        node_logger.error(error_msg)
    else:
        user_q = state["user_text_query"]
        sim_summary = state.get(
            "similar_tracks_summary", "No similarity analysis provided."
        )
        node_logger.info(f"User Query: '{user_q}'")
        node_logger.debug(f"Similarity Summary: {sim_summary[:500]}...")
        prompt = INITIAL_GOAL_SYNTHESIS_PROMPT_TEMPLATE.format(
            user_text_query=user_q, similar_tracks_summary=sim_summary
        )
        
        node_logger.debug("=== INITIAL GOAL SYNTHESIS PROMPT ===")
        node_logger.debug(prompt)
        node_logger.debug("=== END PROMPT ===")

        cb = OpenAICallbackHandler()

        #project goal summary extraction
        def extract_goal_summary(response: str, marker: str) -> str:
            """
            Extracts the final project goal from a CoT-structured response.
            Looks for the marker and returns everything after it.
            Falls back to the full response if marker not found.
            """
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    final_part = parts[1].strip()
                    # Remove any markdown formatting artifacts
                    final_part = final_part.replace("**", "").replace("*", "")
                    return final_part
            
            # Fallback: return the original response
            return response
        # Define the marker for the final project goal

        try:
            node_logger.info("Generating project goal summary...")
            resp = llm.invoke(prompt, config={"callbacks": [cb]})
            raw_goal_response = resp.content.strip()
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            
            node_logger.info("=== PROJECT GOAL SYNTHESIS - FULL LLM RESPONSE ===")
            node_logger.info(raw_goal_response)
            node_logger.info("=== END PROJECT GOAL RESPONSE ===")
            
            # Extract final response from CoT structure if present
            final_marker = "Concise Project Goal Summary:"
            goal_summary = extract_goal_summary(raw_goal_response, final_marker)
            if not goal_summary:
                node_logger.warning(
                    "No valid project goal summary found in LLM response. Using raw response."
                )
                goal_summary = raw_goal_response
            elif len(goal_summary) < 10:
                node_logger.warning(
                    "Project goal summary is too short. Using raw response."
                )
                goal_summary = raw_goal_response
            node_logger.info("=== END EXTRACTED GOAL ===")
            node_logger.info(
                f"Goal summary generated. Tokens P={prompt_tokens}, C={completion_tokens}"
            )
        except Exception as e:
            node_logger.error(f"Goal synthesis failed: {e}", exc_info=True)
            goal_summary = f"Error synthesizing goal: {e}"
            error_msg = str(e)

    should_run_lyrics = any(
        kw in state["user_text_query"].lower() for kw in ("lyric", "vocal")
    )
    return {
        "project_goal_summary": goal_summary,
        "should_run_lyrics_agent": should_run_lyrics,
        "node_prompt_tokens": prompt_tokens,
        "node_completion_tokens": completion_tokens,
        **({"error_message": f"Initial Processing Error: {error_msg}"} if error_msg else {}),
    }


def _specialist_agent_node_logic(
    agent_name: str,
    state: OverallState,
    knowledge_topic: Optional[str],
    prompt_template: str,
    retrieved_chunks_key_in_prompt: str,
    stackexchange_site: Optional[str] = None,
    is_cot_prompt: bool = False
) -> Dict[str, Any]:
    k = agent_name.lower().replace(" ", "_").replace("&", "and")
    advice_k = f"{k}_advice"
    kb_src_k = f"{k}_kb_sources"
    stack_src_k = f"{k}_stack_sources"


    node_logger.info(f"--- NODE START: {agent_name} Agent ---")
    node_logger.info(f"Agent Name: {agent_name}")
    node_logger.info(f"Normalized Key: {k}")
    node_logger.info(f"Knowledge Topic: {knowledge_topic}")
    node_logger.info(f"StackExchange Site: {stackexchange_site}")

    # Early bail on errors
    if state.get("error_message"):
        msg = f"Skipping {agent_name}: prior critical error."
        node_logger.warning(msg)
        node_logger.info(f"--- NODE FINISH: {agent_name} Agent (Early Exit - Error) ---")
        return {
            advice_k: msg, 
            kb_src_k: [], 
            stack_src_k: [],
            f"{k}_se_query_prompt_tokens": 0,
            f"{k}_se_query_completion_tokens": 0,
            f"{k}_final_advice_prompt_tokens": 0,
            f"{k}_final_advice_completion_tokens": 0,
        }
    if not llm:
        msg = f"Error: LLM not initialized for {agent_name}."
        node_logger.error(msg)
        node_logger.info(f"--- NODE FINISH: {agent_name} Agent (Early Exit - No LLM) ---")
        return {
            advice_k: msg, 
            kb_src_k: [], 
            stack_src_k: [],
            f"{k}_se_query_prompt_tokens": 0,
            f"{k}_se_query_completion_tokens": 0,
            f"{k}_final_advice_prompt_tokens": 0,
            f"{k}_final_advice_completion_tokens": 0,
        }

    project_goal = state.get("project_goal_summary", "")
    node_logger.info(f"Project Goal Summary: {project_goal[:200]}...")
    if not project_goal or "Error:" in project_goal.split(".")[0]:
        msg = f"Skipping {agent_name}: invalid project goal."
        node_logger.warning(msg)
        node_logger.info(f"--- NODE FINISH: {agent_name} Agent (Early Exit - Invalid Goal) ---")
        return {
            advice_k: msg, 
            kb_src_k: [], 
            stack_src_k: [],
            f"{k}_se_query_prompt_tokens": 0,
            f"{k}_se_query_completion_tokens": 0,
            f"{k}_final_advice_prompt_tokens": 0,
            f"{k}_final_advice_completion_tokens": 0,
        }

    # Token counters
    se_q_p, se_q_c = 0, 0
    final_p, final_c = 0, 0

    # 1) KB Retrieval
    kb_sources: List[str] = []
    kb_ctx = "No KB info."
    if knowledge_topic:
        node_logger.info(f"Step 1: Retrieving KB knowledge for topic '{knowledge_topic}'...")
        query_text = f"{agent_name} concepts for: {project_goal}"
        node_logger.debug(f"KB Query: '{query_text}'")
        
        docs = get_knowledge_from_kb(
            query_text,
            knowledge_topic,
            DEFAULT_RETRIEVAL_TOP_K,
        )
        if docs:
            kb_sources = [
                f"KB: {d.metadata.get('source','N/A')} (Pg:{d.metadata.get('page_number','N/A')})"
                for d in docs
            ]
            kb_ctx = "\n---\n".join(
                f"{src}:\n{d.page_content}" for src, d in zip(kb_sources, docs)
            )
            node_logger.info(f"Retrieved {len(docs)} KB docs for {agent_name}.")
            node_logger.debug("=== KB RETRIEVAL RESULTS ===")
            for i, (src, doc) in enumerate(zip(kb_sources, docs)):
                node_logger.debug(f"KB Doc {i+1}: {src}")
                node_logger.debug(f"Content preview: {doc.page_content[:300]}...")
            node_logger.debug("=== END KB RESULTS ===")
        else:
            node_logger.warning(f"No KB documents retrieved for {agent_name} with topic '{knowledge_topic}'")

    # 2) StackExchange Retrieval
    stack_sources: List[str] = []
    stack_ctx = "No SE info."
    if stackexchange_site:
        node_logger.info(f"Step 2: Generating StackExchange query for site '{stackexchange_site}'...")
        cb1 = OpenAICallbackHandler()
        gen_q = STACKEXCHANGE_QUESTION_GENERATION_PROMPT_TEMPLATE.format(
            project_goal_summary=project_goal, focus_area=agent_name
        )
        
        node_logger.debug("=== SE QUERY GENERATION PROMPT ===")
        node_logger.debug(gen_q)
        node_logger.debug("=== END PROMPT ===")
        
        resp1 = llm.invoke(gen_q, config={"callbacks": [cb1]})
        #replace possible " " with empty string
        replacements = str.maketrans({'"': '', '“': '', '”': '', '‘': '', '’': ''})
        raw_q = resp1.content.translate(replacements).strip()
        se_q_p, se_q_c = cb1.prompt_tokens, cb1.completion_tokens
        
        node_logger.info("=== STACKEXCHANGE QUERY GENERATION - FULL LLM RESPONSE ===")
        node_logger.info(raw_q)
        node_logger.info("=== END STACKEXCHANGE QUERY RESPONSE ===")
        node_logger.info(f"SE query tokens P={se_q_p}, C={se_q_c}")

        if raw_q:
            node_logger.info(f"Step 2b: Searching StackExchange with query: '{raw_q}'...")
            results = search_stackexchange_qa(raw_q, stackexchange_site, 3, 1)
            if results:
                stack_sources = [
                    f"SE({item['source_url']}): {item['question_title']}"
                    for item in results
                ]
                stack_ctx = "\n=====\n".join(
                    "Q:"
                    + item["question_title"]
                    + "\nA:"
                    + "\n---\n".join(
                        BeautifulSoup(html, "html.parser").get_text(
                            separator="\n", strip=True
                        )
                        for html in item["answer_html_bodies"]
                    )
                    for item in results
                )
                node_logger.info(f"Retrieved {len(results)} SE Q&A for {agent_name}.")
                node_logger.debug("=== SE RETRIEVAL RESULTS ===")
                for i, item in enumerate(results):
                    node_logger.debug(f"SE Q&A {i+1}: {item['question_title']}")
                    node_logger.debug(f"URL: {item['source_url']}")
                    node_logger.debug(f"Answers: {len(item['answer_html_bodies'])}")
                node_logger.debug("=== END SE RESULTS ===")
            else:
                node_logger.warning(f"No StackExchange results retrieved for query: '{raw_q}'")

    # 3) Final Advice Generation
    node_logger.info(f"Step 3: Generating final {agent_name} advice...")
    combined = f"KB INFO:\n{kb_ctx}\n\nSE INFO:\n{stack_ctx}"
    node_logger.debug(f"Combined context length: {len(combined)} characters")
    
    final_prompt = prompt_template.format(
        project_goal_summary=project_goal, **{retrieved_chunks_key_in_prompt: combined}
    )
    
    node_logger.debug("=== FINAL ADVICE GENERATION PROMPT ===")
    node_logger.debug(final_prompt)
    node_logger.debug("=== END PROMPT ===")
    
    cb2 = OpenAICallbackHandler()
    resp2 = llm.invoke(final_prompt, config={"callbacks": [cb2]})
    advice = resp2.content.strip()
    final_p, final_c = cb2.prompt_tokens, cb2.completion_tokens
    
    node_logger.info(f"=== {agent_name.upper()} FINAL ADVICE - FULL LLM RESPONSE ===")
    node_logger.info(advice)
    node_logger.info(f"=== END {agent_name.upper()} FINAL ADVICE RESPONSE ===")
    node_logger.info(f"{agent_name} advice tokens P={final_p}, C={final_c}")



    if is_cot_prompt:
        marker = f"Final {agent_name} Advice:"
        if marker == "Final Lyrics Advice:":
            marker = "Final Lyrics & Vocal Advice:"
        if marker == "Final Production Advice:":
            marker = "Final Production & Mix Advice:"
        if marker == "Final Instruments Advice:":
            marker = "Final Instruments & Timbre Advice:"
        node_logger.info(f"Parsing CoT output for {agent_name} using marker: '{marker}'")

        if marker in advice:
            advice_content = advice.split(marker, 1)[1].strip()
            node_logger.info(f"✅ Successfully parsed CoT output for {agent_name} using marker: '{marker}'")
            node_logger.info(f"=== PARSED {agent_name.upper()} ADVICE (EXTRACTED FROM COT) ===")
            node_logger.info(advice_content)
            node_logger.info(f"=== END PARSED {agent_name.upper()} ADVICE ===")
        else:
            node_logger.warning(f"⚠️ Could not find '{marker}' in CoT output for {agent_name}. Using raw output.")
            advice_content = advice 
    else:
        advice_content = advice  
    advice = advice_content

    node_logger.info(f"🎯 {agent_name} Advice Generation Complete!")
    node_logger.info(f"=== FINAL {agent_name.upper()} ADVICE SUMMARY ===")
    node_logger.info(f"Length: {len(advice)} characters")
    node_logger.info(f"Preview: {advice[:200]}...")
    node_logger.info(f"=== END {agent_name.upper()} ADVICE SUMMARY ===")

    node_logger.debug(f"Final Parsed Advice Snippet for {agent_name}:\n{advice[:200]}...")





    # Debug: Log the exact keys being returned
    result_keys = {
        advice_k: advice,
        kb_src_k: kb_sources,
        stack_src_k: stack_sources,
        f"{k}_se_query_prompt_tokens": se_q_p,
        f"{k}_se_query_completion_tokens": se_q_c,
        f"{k}_final_advice_prompt_tokens": final_p,
        f"{k}_final_advice_completion_tokens": final_c,
    }
    
    token_keys_returned = [key for key in result_keys.keys() if "token" in key]
    node_logger.info(f"{agent_name} returning token keys: {token_keys_returned} with values: {[(key, result_keys[key]) for key in token_keys_returned]}")

    node_logger.info(f"--- NODE FINISH: {agent_name} Agent ---")
    return result_keys


def rhythm_agent_node(state: OverallState) -> Dict[str, Any]:
    return _specialist_agent_node_logic(
        "Rhythm",
        state,
        "rythm",
        RHYTHM_ADVICE_COT_PROMPT_TEMPLATE,
        "retrieved_rhythm_chunks",
        "music.stackexchange.com",
        is_cot_prompt=True,  
    )


def music_theory_agent_node(state: OverallState) -> Dict[str, Any]:
    return _specialist_agent_node_logic(
        "Music Theory",
        state,
        "theory_general",
        MUSIC_THEORY_ADVICE_COT_PROMPT_TEMPLATE,
        "retrieved_music_theory_chunks",
        "music.stackexchange.com",
        is_cot_prompt=True,
    )


def instruments_agent_node(state: OverallState) -> Dict[str, Any]:
    return _specialist_agent_node_logic(
        "Instruments",
        state,
        "timbre_instruments",
        INSTRUMENTS_ADVICE_COT_PROMPT_TEMPLATE,
        "retrieved_instruments_chunks",
        "music.stackexchange.com",
        is_cot_prompt=True,
    )


def lyrics_agent_node(state: OverallState) -> Dict[str, Any]:
    node_logger.info("--- NODE START: Lyrics Agent (conditional) ---")
    if not state.get("should_run_lyrics_agent", False):
        node_logger.info("Skipping Lyrics Agent by flag.")
        return {
            "lyrics_advice": "",
            "lyrics_kb_sources": [],
            "lyrics_stack_sources": [],
            "lyrics_se_query_prompt_tokens": 0,
            "lyrics_se_query_completion_tokens": 0,
            "lyrics_final_advice_prompt_tokens": 0,
            "lyrics_final_advice_completion_tokens": 0,
        }
    return _specialist_agent_node_logic(
        "Lyrics",
        state,
        "Lyrics",
        LYRICS_ADVICE_COT_PROMPT_TEMPLATE,
        "retrieved_lyrics_chunks",
        "writers.stackexchange.com",
        is_cot_prompt=True,
    )


def production_agent_node(state: OverallState) -> Dict[str, Any]:
    return _specialist_agent_node_logic(
        "Production",
        state,
        "production",
        PRODUCTION_ADVICE_COT_PROMPT_TEMPLATE,
        "retrieved_production_chunks",
        "music.stackexchange.com",
        is_cot_prompt=True,
    )


def combine_advice_node(state: OverallState) -> Dict[str, Any]:
    node_logger.info("--- NODE START: Combining Advice ---")
    pg = state.get("project_goal_summary", "N/A")
    err = state.get("error_message")
    parts: List[str] = []
    all_src: List[str] = []

    node_logger.info(f"Project Goal: {pg}")
    if err:
        node_logger.warning(f"Error detected: {err}")

    if err:
        parts.append(f"# FAILED\n{err}\nGoal: {pg}")
    else:
        parts.append(f"# Music Inspiration Moodboard\n## Project Goal\n{pg}\n")

    sections = [
        ("Rhythm & Groove", "rhythm_advice", "rhythm_kb_sources", "rhythm_stack_sources"),
        ("Music Theory & Harmony", "music_theory_advice", "music_theory_kb_sources", "music_theory_stack_sources"),
        ("Timbre & Instrumentation", "instruments_advice", "instruments_kb_sources", "instruments_stack_sources"),
        ("Lyrics & Vocals", "lyrics_advice", "lyrics_kb_sources", "lyrics_stack_sources", True, state.get("should_run_lyrics_agent", False)),
        ("Production & Mix", "production_advice", "production_kb_sources", "production_stack_sources"),
    ]

    node_logger.info("Processing advice sections...")
    for sec in sections:
        title, adv_k, kb_k, stk_k = sec[0], sec[1], sec[2], sec[3]
        cond, ok = (sec[4], sec[5]) if len(sec) > 4 else (False, True)
        
        node_logger.debug(f"Processing section: {title}")
        if cond and not ok:
            node_logger.info(f"Skipping {title} (conditional skip)")
            parts.append(f"## {title}\n(Skipped)")
            continue
            
        adv = state.get(adv_k)
        kbs = state.get(kb_k, [])
        sts = state.get(stk_k, [])
        
        if adv:
            node_logger.info(f"Including {title}: {len(adv)} chars, {len(kbs)} KB sources, {len(sts)} SE sources")
            node_logger.debug(f"{title} advice preview: {adv[:200]}...")
            parts.append(f"## {title}\n{adv}")
            all_src += kbs + sts
        else:
            node_logger.warning(f"No advice found for {title} (key: {adv_k})")

    uniq = sorted(set(all_src))
    if uniq:
        node_logger.info(f"Adding {len(uniq)} unique sources")
        parts.append("## Sources")
        parts += [f"{i+1}. {s}" for i, s in enumerate(uniq)]

    node_logger.info("Calculating total token usage...")
    agent_key_prefixes = ["rhythm", "music_theory", "instruments", "lyrics", "production"]
    total_p = state.get("node_prompt_tokens", 0)
    total_c = state.get("node_completion_tokens", 0)
    
    node_logger.debug(f"Initial tokens from node: P={total_p}, C={total_c}")
    
    token_keys = [k for k in state.keys() if "token" in k]
    node_logger.debug(f"All token-related keys in state: {token_keys}")
    
    for prefix in agent_key_prefixes:
        se_q_p = state.get(f"{prefix}_se_query_prompt_tokens", 0)
        se_q_c = state.get(f"{prefix}_se_query_completion_tokens", 0)
        final_p = state.get(f"{prefix}_final_advice_prompt_tokens", 0)
        final_c = state.get(f"{prefix}_final_advice_completion_tokens", 0)
        
        agent_total_p = se_q_p + final_p
        agent_total_c = se_q_c + final_c
        total_p += agent_total_p
        total_c += agent_total_c
        
        if agent_total_p or agent_total_c:
            node_logger.info(f"{prefix} tokens: SE(P={se_q_p},C={se_q_c}) + Final(P={final_p},C={final_c}) = Total(P={agent_total_p},C={agent_total_c})")
        else:
            node_logger.warning(f"{prefix} tokens: all zero - agent may have failed or keys missing")

    node_logger.info(f"FINAL Total tokens P={total_p}, C={total_c}")
    
    final_moodboard = "\n\n".join(parts)
    node_logger.info(f"Generated moodboard: {len(final_moodboard)} characters total")

    node_logger.info(f"--- NODE FINISH: Combining Advice ---")
    return {
        "final_moodboard": final_moodboard,
        "all_accumulated_sources": uniq,
        "total_prompt_tokens": total_p,
        "total_completion_tokens": total_c,
    }
