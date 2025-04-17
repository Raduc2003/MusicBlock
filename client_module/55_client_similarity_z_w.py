#!/usr/bin/env python3
import os
import json
import sys
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colorama import init, Fore, Style
import argparse
import math # For sqrt

# Initialize colorama
init(autoreset=True)

# --- Configuration (Adjust collection name if needed) ---
COLLECTION_NAME = "music_similarity_zscore2" # Suggest new name
TOP_K = 20
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
STATS_FILE = "global_mean_std.json"
EXPECTED_DIM = 55

# --- Feature Group Indices and Dimensions ---
GROUP_TIMBRE_INDICES = slice(0, 13); N_TIMBRE = 13
GROUP_RHYTHM_INDICES = slice(13, 16); N_RHYTHM = 3
GROUP_TONAL_INDICES = slice(16, 55); N_TONAL = 39
# ----------------------------------------------

# --- Load Stats, Z-Score Norm, Feature Extraction (Keep as before) ---
def load_mean_std_stats(file_path):
    # ... (same as before)
    if not os.path.isfile(file_path): print(f"Error: Stats file not found: {file_path}", file=sys.stderr); sys.exit(1)
    try:
        with open(file_path, "r") as f: stats = json.load(f)
        if len(stats) != EXPECTED_DIM: print(f"Error: Stats file dimensions mismatch.", file=sys.stderr); sys.exit(1)
        return stats
    except Exception as e: print(f"Error loading stats file: {e}", file=sys.stderr); sys.exit(1)

def normalize_vector_zscore(vector, mean_std_stats):
    # ... (same as before)
    normalized = []
    if len(vector) != len(mean_std_stats): raise ValueError("Length mismatch")
    for i, x in enumerate(vector):
        mean, std_dev = mean_std_stats[i]
        normalized_value = 0.0 if std_dev < 1e-9 else (x - mean) / std_dev
        normalized.append(normalized_value)
    return normalized

def extract_features_from_json(json_path, verbose=True):
    # ... (same as before)
    try:
        with open(json_path, "r") as f: data = json.load(f)
    except Exception as e: print(f"Error reading query JSON {json_path}: {e}", file=sys.stderr); sys.exit(1)
    try:
        mfcc_mean = data["lowlevel"]["mfcc"]["mean"]
        bpm = data["rhythm"]["bpm"]; onset_rate = data["rhythm"]["onset_rate"]; danceability = data["rhythm"]["danceability"]
        key_strength = data["tonal"].get("key_strength", data["tonal"].get("key_temperley", {}).get("strength", 0))
        chords_changes_rate = data["tonal"]["chords_changes_rate"]; hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]
        hpcp_mean = data["tonal"]["hpcp"]["mean"]
        feature_vector = ( mfcc_mean + [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] + hpcp_mean )
        if len(feature_vector) != EXPECTED_DIM: raise ValueError("Incorrect feature vector length")
        if verbose: print(f"{Fore.YELLOW}Extracted raw features ({len(feature_vector)} dims)\n", file=sys.stderr)
        return feature_vector
    except Exception as e: print(f"Error extracting features from {json_path}: {e}", file=sys.stderr); sys.exit(1)
# -----------------------------------------------------------------------

# --- NEW: Simpler Scaling Function ---
def apply_direct_scaling(vector, group_weights):
    """
    Applies scaling directly to Z-score features based on group weights
    and dimensionality. Returns numpy array.
    """
    vec_np = np.array(vector, dtype=np.float64)
    scaled_vector = np.zeros_like(vec_np)

    # Calculate scaling factors (adjust this logic as needed)
    # Aim: Make expected squared norm of group proportional to weight
    # scale_factor = sqrt(group_weight / num_dims)
    scale_timbre = math.sqrt(group_weights.get('timbre', 0.0) / N_TIMBRE) if N_TIMBRE > 0 else 0
    scale_rhythm = math.sqrt(group_weights.get('rhythm', 0.0) / N_RHYTHM) if N_RHYTHM > 0 else 0
    scale_tonal = math.sqrt(group_weights.get('tonal', 0.0) / N_TONAL) if N_TONAL > 0 else 0

    # Apply scaling to each group slice
    scaled_vector[GROUP_TIMBRE_INDICES] = vec_np[GROUP_TIMBRE_INDICES] * scale_timbre
    scaled_vector[GROUP_RHYTHM_INDICES] = vec_np[GROUP_RHYTHM_INDICES] * scale_rhythm
    scaled_vector[GROUP_TONAL_INDICES] = vec_np[GROUP_TONAL_INDICES] * scale_tonal

    return scaled_vector
# -------------------------------------

# --- Qdrant Search Function (Keep as before) ---
def perform_similarity_search(query_vector, top_k=TOP_K, collection=COLLECTION_NAME, host=QDRANT_HOST, port=QDRANT_PORT):
    # ... (same as before)
    try:
        client = QdrantClient(host=host, port=port, timeout=20.0)
        try: client.get_collection(collection_name=collection)
        except Exception as e: print(f"Error: Collection '{collection}' not found/conn failed: {e}", file=sys.stderr); sys.exit(1)
        search_result = client.search( collection_name=collection, query_vector=query_vector.tolist(), limit=top_k )
        return search_result
    except Exception as e: print(f"Error during Qdrant search: {e}", file=sys.stderr); sys.exit(1)
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant similarity client using Z-score normalization and direct group scaling.")
    # --- Arguments remain the same, including weights ---
    parser.add_argument("query_json", help="Path to the query JSON file (features).")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Qdrant collection name (default: {COLLECTION_NAME}).")
    parser.add_argument("--stats_file", default=STATS_FILE, help=f"Path to the [mean, std_dev] stats file (default: {STATS_FILE}).")
    parser.add_argument("--host", default=QDRANT_HOST, help=f"Qdrant host (default: read env or '{QDRANT_HOST}').")
    parser.add_argument("--port", type=int, default=QDRANT_PORT, help=f"Qdrant port (default: read env or {QDRANT_PORT}).")
    parser.add_argument("--json", action="store_true", help="Output results in raw JSON format.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help=f"Number of similar items (default: {TOP_K}).")
    parser.add_argument("--verbose", action="store_true", help="Print debug output to stderr.")
    parser.add_argument("--w_timbre", type=float, default=0.3, help="Weight for timbre features (default: 1/3).")
    parser.add_argument("--w_rhythm", type=float, default=0.1, help="Weight for rhythm features (default: 1/3).")
    parser.add_argument("--w_tonal", type=float, default=0.6, help="Weight for tonal features (default: 1/3).")
    # ------------------------------------------------------
    args = parser.parse_args()

    if args.verbose: print(f"{Fore.YELLOW}--- Qdrant Similarity Client (Z-Score + Direct Scaling) ---", file=sys.stderr)
    verbose = args.verbose

    # --- Define Group Weights from Arguments ---
    group_weights = { 'timbre': args.w_timbre, 'rhythm': args.w_rhythm, 'tonal': args.w_tonal }
    if verbose: print(f"{Fore.YELLOW}Using Group Weights: {group_weights}", file=sys.stderr)
    # -----------------------------------------

    # 1) Extract raw features
    raw_vector = extract_features_from_json(args.query_json, verbose=verbose)

    # 2) Load mean/std stats
    mean_std_stats = load_mean_std_stats(args.stats_file)

    # 3) Apply Z-score normalization
    zscore_vector = normalize_vector_zscore(raw_vector, mean_std_stats)

    # 4) Apply Direct Scaling
    final_query_vector_np = apply_direct_scaling(zscore_vector, group_weights)

    if verbose:
         raw_np = np.array(raw_vector); zscore_np = np.array(zscore_vector); final_np = final_query_vector_np
         np.set_printoptions(precision=4, suppress=True)
         print(f"{Fore.YELLOW}--- Vector Stages ---", file=sys.stderr)
         print(f"{Fore.CYAN}Raw Vector (excerpt): {Style.RESET_ALL}{raw_np[:5]}...{raw_np[-5:]}", file=sys.stderr)
         print(f"{Fore.CYAN}Z-Score Vector (excerpt): {Style.RESET_ALL}{zscore_np[:5]}...{zscore_np[-5:]}", file=sys.stderr)
         print(f"{Fore.CYAN}Directly Scaled Query Vector (excerpt): {Style.RESET_ALL}{final_np[:5]}...{final_np[-5:]}", file=sys.stderr)
         final_norm = np.linalg.norm(final_np); print(f"{Fore.YELLOW}Scaled Vector L2 Norm: {final_norm:.4f}", file=sys.stderr)
         print("-" * 20, file=sys.stderr)

    # 5) Perform similarity search using the SCALED vector
    results = perform_similarity_search(
        query_vector=final_query_vector_np, # Pass the numpy array
        top_k=args.top_k,
        collection=args.collection,
        host=args.host,
        port=args.port
    )

    # 6) Output results (remains the same logic)
    if args.json:
        output_data = { "query_scaled_vector": final_query_vector_np.tolist(), "group_weights_used": group_weights, "results": [] }
        for r in results: output_data["results"].append({"id": r.id, "score": r.score, "payload": r.payload})
        print(json.dumps(output_data, indent=2))
    else:
        print(f"\n{Fore.CYAN}{'='*80}"); print(f"{Fore.YELLOW}SIMILARITY SEARCH RESULTS (Z-Score + Directly Scaled Groups)")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Collection: {args.collection}, Top K: {args.top_k}, Weights: {group_weights}")
        print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}\n")
        if not results: print(f"{Fore.RED}No results found.")
        else:
            for i, res in enumerate(results):
                print(f"{Fore.GREEN}Match #{i+1} {Fore.CYAN}{'─'*50}")
                print(f"{Fore.YELLOW}ID: {Fore.WHITE}{res.id}")
                print(f"{Fore.YELLOW}Score (Cosine Sim): {Fore.WHITE}{res.score:.4f}")
                artist = res.payload.get("artist", "N/A"); title = res.payload.get("title", "N/A"); mbid = res.payload.get("mbid", "N/A")
                print(f"{Fore.YELLOW}Artist: {Fore.WHITE}{artist}"); print(f"{Fore.YELLOW}Title: {Fore.WHITE}{title}"); print(f"{Fore.YELLOW}MBID: {Fore.WHITE}{mbid}")
                print("")