# main_orchestrator_api.py
# (Place this in your project root or a dedicated 'orchestrator_service/' directory)

import logging
import os
import shutil # For potentially saving/handling uploaded files if needed locally
from typing import Optional, Dict, Any, List

import fastapi
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field # For request/response models
import uvicorn
import httpx # For making async HTTP requests to other services
import asyncio # For simulating delays or handling async tasks

# --- Configuration ---
# These should be set via environment variables for flexibility in deployment
FEATURE_EXTRACTOR_API_URL = os.getenv("FEATURE_EXTRACTOR_API_URL", "http://localhost:8001/extract_features") # Example
SIMILARITY_API_URL = os.getenv("SIMILARITY_API_URL", "http://localhost:8002/similar_tracks")  # Changed from "find_similar" to "similar_tracks"
RAG_SERVICE_URL = os.getenv("RAG_API_URL", "http://localhost:8000/generate_moodboard")             # Your RAG API
MUSICBRAINZ_API_BASE_URL = "https://musicbrainz.org/ws/2/"
COVER_ART_ARCHIVE_BASE_URL = "https://coverartarchive.org/"
DEFAULT_TOP_K_SIMILARITY = int(os.getenv("DEFAULT_TOP_K_SIMILARITY", 5)) # Default to 5 similar tracks
USER_AGENT = "SimpleMusicEnricher/0.2 (contact@example.com)"

LISTENBRAINZ_PLAYER_TMPL = "https://listenbrainz.org/player?recording_mbids={mbid}"
# Placeholder for similarity data if the Music Similarity service is skipped or fails for POC
SIMILAR_TRACKS_TABLE_PLACEHOLDER = """
| #  | Title                | Artist                          | Key       | BPM | Genres                      | Other tags                          |
| -- | -------------------- | ------------------------------- | --------- | --- | --------------------------- | ----------------------------------- |
| 1  | Placeholder Track 1  | Placeholder Artist A            | C minor   | 120 | electronic, pop             | synth, upbeat                       |
| 2  | Placeholder Track 2  | Placeholder Artist B            | G major   | 90  | acoustic, folk              | guitar, mellow                      |
""" # Shortened for brevity

# --- Logger Setup ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG").upper(), 
                    format='%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s')
logger = logging.getLogger("OrchestratorAPI") # Specific logger name

app = FastAPI(
    title="Music Moodboard Orchestrator API",
    description="Orchestrates calls to music feature extraction, similarity, and RAG services.",
    version="0.1.0"
)

# --- HTTP Client for Service-to-Service Communication ---
SERVICE_CALL_TIMEOUT = 500 
http_client_global = httpx.AsyncClient(timeout=SERVICE_CALL_TIMEOUT,
                                       follow_redirects=True)

@app.on_event("startup")
async def startup_event():
    logger.info("Orchestrator API starting up...")
    logger.info(f"Feature Extractor Service URL: {FEATURE_EXTRACTOR_API_URL}")
    logger.info(f"Similarity Service URL: {SIMILARITY_API_URL}")
    logger.info(f"RAG Service URL: {RAG_SERVICE_URL}")
    # Optionally, add a small health check to downstream services here

@app.on_event("shutdown")
async def shutdown_event():
    await http_client_global.aclose()
    logger.info("Orchestrator API shutting down...")

# --- Pydantic Models for this API's main endpoint ---
class SimilarTrackDetail(BaseModel):
    title: Optional[str] = "Unknown Title"
    artist: Optional[str] = "Unknown Artist"
    mbid: Optional[str] = None
    album_art_url: Optional[str] = None
    listenbrainz_url: Optional[str] = None
    genres: Optional[List[str]] = [] 
    tags: Optional[List[str]] = [] 
    key_signature: Optional[str] = None
    bpm: Optional[float] = None  # Changed from int to float

class OrchestratorFinalResponse(BaseModel):
    final_moodboard: Optional[str]  # Change this field name
    processed_similar_tracks: Optional[List[SimilarTrackDetail]]
    rag_sources: Optional[List[str]]
    token_usage: Optional[Dict[str, Optional[int]]]
    error_message: Optional[str]
    debug_info: Optional[Dict[str, Any]] = None

class SimilarityRequest(BaseModel):
    feature_json_content: Dict[str, Any]  # Not "features"
    top_k: Optional[int] = Field(DEFAULT_TOP_K_SIMILARITY, ge=1, le=50)


# --- Helper Functions for Orchestrator ---

async def call_feature_extractor_api(audio_content: bytes, audio_filename: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Calling Feature Extractor API for: {audio_filename}")
    files = {'audio_file': (audio_filename, audio_content, 'audio/mpeg')} # Adjust content-type if needed (e.g., audio/wav)
    try:
        response = await http_client_global.post(FEATURE_EXTRACTOR_API_URL, files=files)
        response.raise_for_status()
        features_json = response.json()
        logger.info("Features extracted successfully from API.")
        logger.debug(f"Feature Extractor Response (keys): {list(features_json.keys())}")
        return features_json
    except httpx.RequestError as e:
        logger.error(f"Network error calling Feature Extractor API: {e}", exc_info=True)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Feature Extractor API: {e.response.status_code} - {e.response.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in call_feature_extractor_api: {e}", exc_info=True)
    return None

async def call_similarity_api(raw_features_json: Dict[str, Any], top_k: int = 30) -> Optional[List[Dict[str, Any]]]:
    logger.info(f"Calling Similarity API with top_k={top_k}")
    # Fix: Use the correct field name that matches SimilarityRequest model
    payload = {"feature_json_content": raw_features_json, "top_k": top_k}  # Changed "features" to "feature_json_content"
    try:
        response = await http_client_global.post(SIMILARITY_API_URL, json=payload)
        response.raise_for_status()
        similarity_response = response.json()
        # Fix: Use the correct response field name from your similarity API
        similar_songs = similarity_response.get("similar_tracks", [])  # Changed from "similar_songs" to "similar_tracks"
        logger.info(f"Similar songs retrieved from API: {len(similar_songs)} songs.")
        return similar_songs
    except httpx.RequestError as e:
        logger.error(f"Network error calling Similarity API: {e}", exc_info=True)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from Similarity API: {e.response.status_code} - {e.response.text}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error in call_similarity_api: {e}", exc_info=True)
    return None

def remove_duplicates(songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate songs based on title and artist."""
    seen = set()
    unique_songs = []
    for song in songs:
        identifier = (song.get('title', '').lower(), song.get('artist_name', '').lower())
        if identifier not in seen:
            seen.add(identifier)
            unique_songs.append(song)
    return unique_songs

async def enrich_song(song: Dict[str, Any]) -> Dict[str, Any]:
    """Augment *song* dict with MusicBrainz/CAA metadata and a ListenBrainz link.

    Keys always present in the returned dict:
    - full_title, artist_name, length_ms, isrcs, genres, tags
    - album_title, release_date, release_country, release_status, release_type
    - album_art_url, listenbrainz_url
    """

    mbid = song.get("mbid")
    enriched: Dict[str, Any] = song.copy()

    # --- 1. deterministic schema -------------------------------------------------
    defaults = {
        "full_title": song.get("title"),
        "artist_name": song.get("artist"),
        "length_ms": None,
        "isrcs": [],
        "genres": [],
        "tags": [],
        "album_title": None,
        "release_date": None,
        "release_country": None,
        "release_status": None,
        "release_type": None,
        "album_art_url": "https://via.placeholder.com/250?text=No+Art",
        "listenbrainz_url": None,
    }
    for k, v in defaults.items():
        enriched.setdefault(k, v)

    # ListenBrainz link requires only MBID.
    if mbid:
        enriched["listenbrainz_url"] = LISTENBRAINZ_PLAYER_TMPL.format(mbid=mbid)
    else:
        logger.debug("No MBID for %s – skipping enrichment", song.get("title"))
        return enriched

    # --- 2. MusicBrainz request ---------------------------------------------------
    rec_url = (
        f"{MUSICBRAINZ_API_BASE_URL}recording/{mbid}"
        "?fmt=json&inc="
        "artist-credits+releases+release-groups+isrcs+genres+tags"
    )
    try:
        resp = await http_client_global.get(rec_url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("MusicBrainz lookup failed for %s: %s", mbid, exc)
        return enriched

    # --- 3. Basic recording fields ----------------------------------------------
    enriched["full_title"] = data.get("title", enriched["full_title"])
    enriched["length_ms"] = data.get("length")
    enriched["isrcs"] = data.get("isrcs", [])

    # --- 4. Collect genres & tags -------------------------------------------------
    genre_set: Set[str] = {g["name"] for g in data.get("genres", [])}
    tag_set: Set[str] = {t["name"] for t in data.get("tags", [])}

    for rel in data.get("releases", []):
        genre_set.update(g["name"] for g in rel.get("genres", []))
        tag_set.update(t["name"] for t in rel.get("tags", []))

    if not genre_set or not tag_set:
        if data.get("artist-credit"):
            artist_obj = data["artist-credit"][0].get("artist", {})
            genre_set.update(g["name"] for g in artist_obj.get("genres", []))
            tag_set.update(t["name"] for t in artist_obj.get("tags", []))

    enriched["genres"] = sorted(genre_set)
    enriched["tags"] = sorted(tag_set)

    # --- 5. Release‑level details -------------------------------------------------
    primary_release: Optional[Dict[str, Any]] = data.get("releases", [{}])[0] if data.get("releases") else None
    if primary_release:
        enriched["album_title"] = primary_release.get("title")
        enriched["release_date"] = primary_release.get("date")
        enriched["release_country"] = primary_release.get("country")
        enriched["release_status"] = primary_release.get("status")
        rg = primary_release.get("release-group")
        enriched["release_type"] = rg.get("primary-type") if rg else primary_release.get("status")

    # --- 6. Cover art -------------------------------------------------------------
    art_url: Optional[str] = None

    # 6a. Check every release for available art (first success wins)
    for rel in data.get("releases", []):
        rid = rel.get("id")
        if not rid:
            continue
        candidate = f"{COVER_ART_ARCHIVE_BASE_URL}/release/{rid}/front-250"
        try:
            head = await http_client_global.head(candidate, follow_redirects=True)
            if head.status_code == 200:
                art_url = candidate
                break
        except Exception:
            continue

    # 6b. If none of the releases had art, iterate through release‑groups
    if not art_url:
        for rg in data.get("release-groups", []):
            rgid = rg.get("id")
            if not rgid:
                continue
            candidate = f"{COVER_ART_ARCHIVE_BASE_URL}/release-group/{rgid}/front-250"
            try:
                head = await http_client_global.head(candidate, follow_redirects=True)
                if head.status_code == 200:
                    art_url = candidate
                    break
            except Exception:
                continue

    if art_url:
        enriched["album_art_url"] = art_url

    # --- 7. Artist name override --------------------------------------------------
    if data.get("artist-credit"):
        enriched["artist_name"] = data["artist-credit"][0].get("name", enriched["artist_name"])

    return enriched


def generate_rag_summary_from_songs(enriched_songs: List[Dict[str, Any]]) -> str:
    logger.info(f"Generating RAG summary from {len(enriched_songs)} enriched songs...")
    if not enriched_songs:
        return "No specific similar tracks were identified to inform the RAG context."

    summary_parts = ["Key characteristics from acoustically similar tracks for context:\n"]
    
    for i, song in enumerate(enriched_songs): # Summarize all for RAG prompt
        logger.debug(f"full song data: {song}")
        # Use get() to avoid KeyError if keys are missing
        if not song:
            logger.warning(f"Skipping empty song data at index {i}")
            continue
        title = song.get('full_title', song.get('title', 'N/A'))
        artist = song.get('artist_name', song.get('artist', 'N/A'))
        genres = ", ".join(song.get('genres', [])) if song.get('genres') else 'N/A'
        tags = ", ".join(song.get('tags', [])) if song.get('tags') else 'N/A'
        key = song.get('key', song.get('key', 'N/A'))
        bpm = song.get('bpm', 'N/A')
        summary_parts.append(f"{i+1}. '{title}' by {artist} (Genres: {genres}; Key: {key}; BPM: {bpm}; Tags: {tags})")

    summary = "\n".join(summary_parts)
    logger.debug(f"Generated RAG summary:\n{summary}")
    return summary

# --- API Endpoint ---
@app.post("/create_inspiration_moodboard", response_model=OrchestratorFinalResponse)
async def main_moodboard_endpoint(
    user_text_query: str = Form(...),
    audio_file: Optional[UploadFile] = File(None)
):
    logger.info(f"Orchestrator main endpoint hit. User Query: '{user_text_query[:50]}...' Audio: {'Yes' if audio_file else 'No'}")
    
    similar_tracks_summary_for_rag: str
    processed_similar_tracks_for_client: List[SimilarTrackDetail] = []
    debug_info = {}

    # Fix the audio file detection
    if audio_file and audio_file.filename and audio_file.size > 0:  # Add size check
        audio_contents = await audio_file.read()
        audio_filename = audio_file.filename
        await audio_file.close()
        logger.info(f"Audio file '{audio_filename}' received ({len(audio_contents)} bytes). Processing through similarity pipeline...")

        raw_features_json = await call_feature_extractor_api(audio_contents, audio_filename)
        if not raw_features_json:
            logger.warning("Feature extraction failed. Using placeholder similarity summary for RAG.")
            similar_tracks_summary_for_rag = f"Audio feature extraction failed. Using general context:\n{SIMILAR_TRACKS_TABLE_PLACEHOLDER}"
            debug_info["similarity_pipeline_status"] = "Feature extraction failed"
        else:
            logger.info("Feature extraction successful. Calling similarity API...")
            # Assuming similarity_api expects the whole JSON, adapt if it needs a specific key from it
            similar_songs_raw_list_full = await call_similarity_api(raw_features_json) 
            
            # Remove duplicates based on title and artist

            similar_songs_raw_list = remove_duplicates(similar_songs_raw_list_full) if similar_songs_raw_list_full else None
            logger.debug(f"Raw similar songs list after deduplication: {similar_songs_raw_list}")

            if similar_songs_raw_list is None:
                logger.warning("Similarity API call failed. Using placeholder summary for RAG.")
                similar_tracks_summary_for_rag = f"Similarity API failed. Using general context:\n{SIMILAR_TRACKS_TABLE_PLACEHOLDER}"
                debug_info["similarity_pipeline_status"] = "Similarity API call failed"
            elif not similar_songs_raw_list:
                logger.info("No closely similar tracks found by similarity API.")
                similar_tracks_summary_for_rag = "No closely similar tracks found based on audio analysis. Broadening context for RAG."
                debug_info["similarity_pipeline_status"] = "No similar tracks found"
            else:
                logger.info(f"Found {len(similar_songs_raw_list)} raw similar songs. Enriching details...")
                enriched_songs_data = []
                for song_raw in similar_songs_raw_list:
                    enriched_song = await enrich_song(song_raw)
                    enriched_songs_data.append(enriched_song)
                
                similar_tracks_summary_for_rag = generate_rag_summary_from_songs(enriched_songs_data)
                # Prepare the list for client display using Pydantic model
                processed_similar_tracks_for_client = []
                for song in enriched_songs_data:
                    try:
                        # Convert int bpm to float if needed, or keep as is if already float/None
                        bpm_value = song.get('bpm')
                        if isinstance(bpm_value, (int, float)):
                            bpm_value = float(bpm_value)
                        
                        processed_similar_tracks_for_client.append(SimilarTrackDetail(
                            title=song.get('full_title', song.get('title')),
                            artist=song.get('artist_name', song.get('artist')),
                            mbid=song.get('mbid'),
                            album_art_url=song.get('album_art_url'),
                            listenbrainz_url=song.get('listenbrainz_url'),
                            genres=song.get('genres', []),
                            tags=song.get('tags', []),
                            key_signature=song.get('key', None),
                            bpm=bpm_value
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to process enriched song {song.get('title', 'unknown')}: {e}")
                        continue

                debug_info["similarity_pipeline_status"] = "Success"
                debug_info["enriched_similar_tracks_count"] = len(processed_similar_tracks_for_client)
    else:
        logger.info("No audio file provided by user. Using placeholder similarity summary for RAG.")
        similar_tracks_summary_for_rag = f"No audio provided. General inspirational context:\n{SIMILAR_TRACKS_TABLE_PLACEHOLDER}"
        debug_info["similarity_pipeline_status"] = "No audio provided by user"
        # Optionally populate processed_similar_tracks_for_client with placeholder data
        processed_similar_tracks_for_client = [
            SimilarTrackDetail(title="Placeholder Track 1", artist="Placeholder Artist A", genres=["electronic", "pop"]),
            SimilarTrackDetail(title="Placeholder Track 2", artist="Placeholder Artist B", genres=["acoustic", "folk"]),
        ]


    # Call RAG API Service
    rag_payload = {
        "user_text_query": user_text_query,
        "similar_tracks_summary": similar_tracks_summary_for_rag
    }
    logger.info(f"Calling RAG service with RAG summary (snippet): {similar_tracks_summary_for_rag[:100]}...")
    logger.debug(f"Full RAG API Payload: {rag_payload}")

    try:
        rag_response = await http_client_global.post(RAG_SERVICE_URL, json=rag_payload)
        rag_response.raise_for_status()
        rag_data = rag_response.json()
        logger.info("Successfully received response from RAG service.")
    except Exception as e_rag:
        logger.error(f"Error calling RAG service: {e_rag}", exc_info=True)
        error_detail = str(e_rag)
        if isinstance(e_rag, httpx.HTTPStatusError): error_detail = e_rag.response.text
        return OrchestratorFinalResponse( # Return error within the defined response model
            final_moodboard=None,  # Use correct field name
            processed_similar_tracks=processed_similar_tracks_for_client,
            rag_sources=[],
            token_usage={},
            error_message=f"RAG service error: {error_detail}",
            debug_info=debug_info
        )

    return OrchestratorFinalResponse(
        final_moodboard=rag_data.get("final_moodboard"),  # Use correct field name
        processed_similar_tracks=processed_similar_tracks_for_client,
        rag_sources=rag_data.get("all_accumulated_sources"),
        token_usage={
            "prompt_tokens": rag_data.get("total_prompt_tokens"),
            "completion_tokens": rag_data.get("total_completion_tokens")
        },
        error_message=rag_data.get("error_message"),
        debug_info=debug_info
    )

if __name__ == "__main__":
    port = int(os.getenv("ORCHESTRATOR_PORT", 8080))

    logger.info(f"Starting Main Orchestrator API server with Uvicorn on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")