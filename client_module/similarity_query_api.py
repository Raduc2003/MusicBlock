# similarity_pipeline/similarity_query_api.py

import logging
import os
import sys
import json
from typing import List, Dict, Optional, Any # Add Any

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field # conlist removed as input is now Dict
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MVP Configuration
EXPECTED_DIM = 94
DEFAULT_SIMILARITY_COLLECTION_NAME = os.getenv("SIMILARITY_COLLECTION_NAME", "zscore_94")
DEFAULT_STATS_FILE_PATH_IN_CONTAINER = os.getenv("STATS_FILE_PATH", "/app/global_mean_std_94FEATURES.json")
DEFAULT_TOP_K_SIMILARITY = int(os.getenv("TOP_K_SIMILARITY", 10))
QDRANT_HOST_SIMILARITY = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT_SIMILARITY = int(os.getenv("QDRANT_PORT", 6333))

# Load normalization statistics
MEAN_STD_STATS_GLOBAL: Optional[List[List[float]]] = None
try:
    with open(DEFAULT_STATS_FILE_PATH_IN_CONTAINER, "r") as f:
        stats_data = json.load(f)
    if len(stats_data) != EXPECTED_DIM:
        raise ValueError(f"Stats dimensions mismatch. Expected {EXPECTED_DIM}, found {len(stats_data)}")
    MEAN_STD_STATS_GLOBAL = [[float(mean), float(std)] for mean, std in stats_data]
    logger.info(f"Loaded normalization stats: {len(MEAN_STD_STATS_GLOBAL)} dimensions")
except Exception as e:
    logger.error(f"Failed to load stats file: {e}")
    MEAN_STD_STATS_GLOBAL = None


# --- Feature Extraction from JSON content (adapted from your script) ---
def extract_features_from_json_content(data: Dict[str, Any], source_identifier: str = "api_input") -> List[float]:
    """Extract 94-dimensional feature vector from Essentia JSON"""
    try:
        feature_vector = []
        low = data["lowlevel"]
        rhy = data["rhythm"]
        ton = data["tonal"]

        # MFCC features (24 dims: 12 mean + 12 std)
        mfcc_mean = low["mfcc"]["mean"]
        if len(mfcc_mean) < 13: raise ValueError("MFCC mean length < 13")
        feature_vector.extend(mfcc_mean[1:13])

        mfcc_cov = low["mfcc"]["cov"]
        mfcc_variances = np.diag(mfcc_cov)
        if len(mfcc_variances) < 13: raise ValueError("MFCC cov diagonal length < 13")
        mfcc_std = np.sqrt(np.maximum(0, mfcc_variances[1:13]))
        feature_vector.extend(mfcc_std)

        # Spectral features (8 dims)
        feature_vector.append(low["spectral_centroid"]["mean"])
        feature_vector.append(np.sqrt(max(0, low["spectral_centroid"]["var"])))
        feature_vector.append(low["spectral_flux"]["mean"])
        feature_vector.append(np.sqrt(max(0, low["spectral_flux"]["var"])))
        feature_vector.append(low["barkbands_flatness_db"]["mean"])
        feature_vector.append(low["spectral_entropy"]["mean"])
        feature_vector.append(low["zerocrossingrate"]["mean"])
        feature_vector.append(np.sqrt(max(0, low["zerocrossingrate"]["var"])))
        
        # Additional spectral features (4 dims)
        feature_vector.append(low["dissonance"]["mean"])
        feature_vector.append(np.sqrt(max(0, low["dissonance"]["var"])))
        feature_vector.append(low["pitch_salience"]["mean"])
        feature_vector.append(np.sqrt(max(0, low["pitch_salience"]["var"])))
        
        # Spectral contrast (6 dims)
        contrast_coeffs_mean = low["spectral_contrast_coeffs"]["mean"]
        if len(contrast_coeffs_mean) < 6: raise ValueError("Spectral Contrast Coeffs length < 6")
        feature_vector.extend(contrast_coeffs_mean)

        # Rhythm features (13 dims)
        feature_vector.append(rhy["bpm"])
        feature_vector.append(rhy["onset_rate"])
        feature_vector.append(rhy["danceability"])
        pulse_clarity = 0.0
        peak_weight_value = rhy.get("bpm_histogram_first_peak_weight")
        if isinstance(peak_weight_value, dict): pulse_clarity = float(peak_weight_value.get("mean", 0.0))
        elif isinstance(peak_weight_value, (float, int)): pulse_clarity = float(peak_weight_value)
        feature_vector.append(pulse_clarity)
        band_ratio_mean = rhy["beats_loudness_band_ratio"]["mean"]
        if len(band_ratio_mean) < 6: raise ValueError("Band Ratio length < 6")
        feature_vector.extend(band_ratio_mean)
        feature_vector.append(low["dynamic_complexity"])
        feature_vector.append(rhy["beats_count"])
        beats_pos = rhy["beats_position"]
        beat_interval_std = np.std(np.diff(beats_pos)) if len(beats_pos) > 1 else 0.0
        feature_vector.append(beat_interval_std)

        # Tonal features (39 dims: 36 HPCP + 3 key features)
        hpcp_mean = ton["hpcp"]["mean"]
        if len(hpcp_mean) < 36: raise ValueError("HPCP Mean length < 36")
        feature_vector.extend(hpcp_mean)
        feature_vector.append(ton["hpcp_entropy"]["mean"])
        feature_vector.append(ton.get("key_strength", ton.get("key_temperley", {}).get("strength", 0.0)))
        key_scale_str = ton.get("key_scale", ton.get("key_temperley", {}).get("scale", "major"))    
        feature_vector.append(1.0 if key_scale_str == "major" else 0.0)

        # Validation
        if len(feature_vector) != EXPECTED_DIM:
            raise ValueError(f"Extracted feature vector length {len(feature_vector)} != expected {EXPECTED_DIM}")
        for i, x in enumerate(feature_vector):
            if not isinstance(x, (int, float, np.number)): 
                raise TypeError(f"Non-numeric value at index {i}: {x}")
            if not np.isfinite(x): 
                raise ValueError(f"Non-finite value at index {i}: {x}")
        
        return feature_vector
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Feature extraction failed: {e}")

# --- Z-Score Normalization (from your script - unchanged) ---
def normalize_vector_zscore(vector: List[float], mean_std_stats: List[List[float]]) -> List[float]:
    # ... (as before) ...
    normalized = []
    if len(vector) != len(mean_std_stats):
        raise ValueError(f"Vector length {len(vector)} != stats length {len(mean_std_stats)}")
    for i, x_val in enumerate(vector):
        mean, std_dev = mean_std_stats[i]
        if abs(std_dev) < 1e-9: normalized_value = 0.0
        else: normalized_value = (x_val - mean) / std_dev
        normalized.append(normalized_value)
    return normalized

# --- Qdrant Search Function (from your script - complete implementation) ---
def perform_qdrant_similarity_search(
    normalized_query_vector: List[float],
    top_k: int = DEFAULT_TOP_K_SIMILARITY,
    collection_name: str = DEFAULT_SIMILARITY_COLLECTION_NAME,
    host: str = QDRANT_HOST_SIMILARITY,
    port: int = QDRANT_PORT_SIMILARITY
) -> List[qdrant_models.ScoredPoint]:
    """Perform similarity search in Qdrant"""
    if len(normalized_query_vector) != EXPECTED_DIM:
        raise ValueError(f"Query vector length {len(normalized_query_vector)} != expected {EXPECTED_DIM}")

    try:
        client = QdrantClient(host=host, port=port, timeout=20.0)
        search_result = client.search(
            collection_name=collection_name,
            query_vector=normalized_query_vector,
            limit=top_k,
            with_payload=True 
        )
        return search_result
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant search failed: {e}")


# Simplified MVP Models  
class SimilarityRequest(BaseModel):
    feature_json_content: Dict[str, Any] = Field(..., description="Essentia feature JSON content")
    top_k: Optional[int] = Field(DEFAULT_TOP_K_SIMILARITY, ge=1, le=50, description="Number of similar tracks to return")

# Simplified MVP Models
class TrackResult(BaseModel):
    id: Any = Field(..., description="ID of the similar track")
    mbid: Optional[str] = Field(None, description="MusicBrainz ID from payload")
    score: float = Field(..., description="Similarity score (0-1, higher = more similar)")
    title: Optional[str] = None
    artist: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None 

class SimilarityResponse(BaseModel):
    similar_tracks: List[TrackResult]
    total_results: int = Field(..., description="Number of results returned")


app = FastAPI(
    title="Similarity Query API - MVP",
    description="Simplified music similarity search API - extracts features from JSON and finds similar tracks",
    version="1.0.0-mvp"
)

@app.on_event("startup")
async def api_startup():
    """Startup checks for MVP API"""
    if MEAN_STD_STATS_GLOBAL is None:
        logger.critical("Stats file not loaded - API will not work")
    
    # Quick Qdrant connection test
    try:
        client = QdrantClient(host=QDRANT_HOST_SIMILARITY, port=QDRANT_PORT_SIMILARITY, timeout=5.0)
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if DEFAULT_SIMILARITY_COLLECTION_NAME in collection_names:
            logger.info(f"Qdrant connection OK - target collection '{DEFAULT_SIMILARITY_COLLECTION_NAME}' found")
        else:
            logger.warning(f"Target collection '{DEFAULT_SIMILARITY_COLLECTION_NAME}' not found")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")

@app.post("/similar_tracks", response_model=SimilarityResponse)
async def query_similar_tracks_endpoint(request_data: SimilarityRequest):
    """Simplified similarity search endpoint"""
    logger.info(f"Processing similarity query for top_k: {request_data.top_k}")

    if MEAN_STD_STATS_GLOBAL is None:
        raise HTTPException(status_code=500, detail="Stats not loaded")

    # Extract and normalize features
    try:
        raw_vector = extract_features_from_json_content(request_data.feature_json_content)
        normalized_vector = normalize_vector_zscore(raw_vector, MEAN_STD_STATS_GLOBAL)
        logger.info(f"Feature extraction and normalization complete")
    except Exception as e:
        logger.error(f"Feature processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Feature processing failed: {str(e)}")

    # Perform similarity search
    try:
        qdrant_results = perform_qdrant_similarity_search(
            normalized_query_vector=normalized_vector,
            top_k=request_data.top_k or DEFAULT_TOP_K_SIMILARITY
        )
        logger.info(f"Found {len(qdrant_results)} similar tracks")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=503, detail="Search failed")

    # Format simplified results
    track_results = []
    for hit in qdrant_results:
        payload = hit.payload or {}
        track = TrackResult(
            id=hit.id,
            score=round(hit.score, 4),
            title=payload.get("title"),
            mbid=payload.get("mbid"),
            artist=payload.get("artist"),
            bpm=payload.get("bpm"),
           
            key=payload.get("key_key") + "_" + payload.get("key_scale") if "key_key" in payload and "key_scale" in payload else None,
        )
        track_results.append(track)
    
    return SimilarityResponse(
        similar_tracks=track_results,
        total_results=len(track_results)
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "version": "1.0.0-mvp"}

@app.get("/debug/qdrant")
async def debug_qdrant():
    """Debug endpoint to check Qdrant connection"""
    try:
        client = QdrantClient(host=QDRANT_HOST_SIMILARITY, port=QDRANT_PORT_SIMILARITY, timeout=5.0)
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        target_exists = DEFAULT_SIMILARITY_COLLECTION_NAME in collection_names
        
        return {
            "connection_status": "success",
            "target_collection": DEFAULT_SIMILARITY_COLLECTION_NAME,
            "target_exists": target_exists,
            "available_collections": collection_names
        }
    except Exception as e:
        return {
            "connection_status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    logger.info("Starting Similarity Query API - MVP version on port 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")