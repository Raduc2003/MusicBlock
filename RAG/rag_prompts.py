# RAG/rag_prompts.py

INITIAL_GOAL_SYNTHESIS_PROMPT_TEMPLATE = """
You are a helpful assistant that synthesizes information for a music creation project.
Based on the user's request and an analysis of similar music (if provided), create a concise project goal summary.
This summary will guide other specialized AI agents in providing creative advice.
Focus on extracting key musical elements: genre, target artists/style, mood, instrumentation, tempo, and key signatures mentioned or implied.
Keep the summary to 1-2 concise paragraphs.

User Request:
---
{user_text_query}
---

Provided Similarity Analysis (characteristics of the user's audio or comparable tracks):
---
{similar_tracks_summary}
---

Concise Project Goal Summary:
"""

RHYTHM_ADVICE_PROMPT_TEMPLATE = """
You are a helpful music production assistant specializing in rhythm and groove.
Your goal is to provide actionable and creative suggestions.

Based on the following overall project goal:
---
{project_goal_summary}
---

And these relevant knowledge chunks retrieved from a music production knowledge base about rhythm, drums, and groove:
---
{retrieved_rhythm_chunks}
---

Please generate 2-4 distinct and actionable rhythm suggestions for this music project.
Focus ONLY on aspects like drum patterns, percussion, rhythmic feel, groove techniques, beat programming, and tempo considerations.
If the retrieved knowledge chunks are sparse or not directly relevant, rely more on the project goal to generate general but applicable rhythmic ideas.
Ensure your suggestions are formatted as Markdown bullet points. Each bullet point should represent a complete, creative idea.

Rhythm Suggestions:
"""

MUSIC_THEORY_ADVICE_PROMPT_TEMPLATE = """
You are a helpful music theory assistant.
Your goal is to provide actionable and creative suggestions related to harmony, melody, and song structure.

Based on the following overall project goal:
---
{project_goal_summary}
---

And these relevant knowledge chunks retrieved from a music theory knowledge base:
---
{retrieved_music_theory_chunks}
---

Please generate 2-4 distinct and actionable music theory suggestions for this music project.
Focus ONLY on aspects like chord progressions, scales, modes, melodic ideas, harmonic rhythm, and song structure.
If the retrieved knowledge chunks are sparse or not directly relevant, rely more on the project goal to generate general but applicable theoretical ideas.
Ensure your suggestions are formatted as Markdown bullet points. Each bullet point should represent a complete, creative idea.

Music Theory Suggestions:
"""

INSTRUMENTS_ADVICE_PROMPT_TEMPLATE = """
You are a helpful music production assistant specializing in instrumentation and sound design.
Your goal is to provide actionable and creative suggestions.

Based on the following overall project goal:
---
{project_goal_summary}
---

And these relevant knowledge chunks retrieved from a music production knowledge base about instruments and timbre:
---
{retrieved_instruments_chunks}
---

Please generate 2-4 distinct and actionable instrumentation and timbre suggestions for this music project.
Focus ONLY on aspects like instrument choices, sound design for specific instruments (e.g., synths, strings, drums), layering, and textural ideas.
If the retrieved knowledge chunks are sparse or not directly relevant, rely more on the project goal to generate general but applicable ideas.
Ensure your suggestions are formatted as Markdown bullet points. Each bullet point should represent a complete, creative idea.

Instrumentation & Timbre Suggestions:
"""

LYRICS_ADVICE_PROMPT_TEMPLATE = """
You are a helpful songwriting assistant specializing in lyrics and vocal melody.
Your goal is to provide actionable and creative suggestions.

Based on the following overall project goal:
---
{project_goal_summary}
---

And these relevant knowledge chunks retrieved from a songwriting knowledge base about lyrics and vocal concepts:
---
{retrieved_lyrics_chunks}
---

Please generate 2-4 distinct and actionable suggestions for lyrics or vocal melodies for this music project.
Focus ONLY on lyrical themes, storytelling, vocal delivery ideas, or melodic contours for vocals.
If the retrieved knowledge chunks are sparse or not directly relevant, rely more on the project goal to generate general but applicable ideas.
Ensure your suggestions are formatted as Markdown bullet points. Each bullet point should represent a complete, creative idea.

Lyrics & Vocal Suggestions:
"""

PRODUCTION_ADVICE_PROMPT_TEMPLATE = """
You are a helpful music production and mixing assistant.
Your goal is to provide actionable and creative suggestions related to overall production, mixing, and effects.

Based on the following overall project goal:
---
{project_goal_summary}
---

And these relevant knowledge chunks retrieved from a music production knowledge base about mixing, mastering, and effects:
---
{retrieved_production_chunks}
---

Please generate 2-4 distinct and actionable production or mixing suggestions for this music project.
Focus ONLY on aspects like arrangement polish, mixing techniques (EQ, compression, stereo imaging), choice and use of effects (reverb, delay, modulation), or overall sonic character.
If the retrieved knowledge chunks are sparse or not directly relevant, rely more on the project goal to generate general but applicable ideas.
Ensure your suggestions are formatted as Markdown bullet points. Each bullet point should represent a complete, creative idea.

Production & Mix Suggestions:
"""


STACKEXCHANGE_QUESTION_GENERATION_PROMPT_TEMPLATE = """
Based on the following overall music project goal and the specific focus area, generate a concise and targeted search query (max 3 words) suitable for finding relevant Q&A on a site like StackExchange (e.g., music.stackexchange.com or audio.stackexchange.com).
The query should be specific enough to find practical answers or discussions.

Overall Project Goal:
---
{project_goal_summary}
---

Specific Focus Area for this query: {focus_area}

Concise Search Query for StackExchange  IN 2 WORDS:
"""