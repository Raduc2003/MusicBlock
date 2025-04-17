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

# --- Configuration ---
# Updated for the 94-dimensional feature set
EXPECTED_DIM = 94
DEFAULT_COLLECTION_NAME = f"zscore_94" # Suggest descriptive name
DEFAULT_STATS_FILE = f"global_mean_std_94FEATURES.json" # Match stats script output
TOP_K = 20
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost") # Changed default to localhost
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
# -----------------------------------------------------

# --- Load Stats ---
def load_mean_std_stats(file_path):
    if not os.path.isfile(file_path):
        print(f"{Fore.RED}Error: Stats file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path, "r") as f:
            stats = json.load(f)
        # Ensure stats is a list of lists/tuples
        if not isinstance(stats, list) or not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in stats):
             raise TypeError("Stats file should contain a list of [mean, std_dev] pairs.")
        if len(stats) != EXPECTED_DIM:
             raise ValueError(f"Stats file dimensions mismatch. Expected {EXPECTED_DIM}, found {len(stats)}.")
        # Convert to float, handle potential None values robustly if needed (though stats file shouldn't have them)
        return [[float(mean), float(std)] for mean, std in stats]
    except Exception as e:
        print(f"{Fore.RED}Error loading or parsing stats file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

# --- Z-Score Normalization ---
def normalize_vector_zscore(vector, mean_std_stats):
    """Applies Z-score normalization using pre-calculated stats."""
    normalized = []
    if len(vector) != len(mean_std_stats):
        # This check should ideally happen after extraction confirms EXPECTED_DIM
        raise ValueError(f"Vector length {len(vector)} != stats length {len(mean_std_stats)}")
    for i, x in enumerate(vector):
        mean, std_dev = mean_std_stats[i]
        # Handle near-zero std dev to prevent division by zero
        if abs(std_dev) < 1e-9:
            normalized_value = 0.0 # Feature is constant, z-score is 0
        else:
            normalized_value = (x - mean) / std_dev
        normalized.append(normalized_value)
    return normalized # Returns a list

# --- Feature Extraction (Copied & adapted from stats script) ---
def extract_features_from_json(json_path, verbose=True):
    """
    Extracts the REVISED ~94 features from a query JSON file.
    Calculates MFCC std dev from covariance. Omits delta features.
    Returns the raw feature vector as a list, or exits on error.
    """
    if verbose: print(f"{Fore.YELLOW}Extracting features from query file: {json_path}", file=sys.stderr)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error reading query JSON {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        feature_vector = []
        low = data["lowlevel"]
        rhy = data["rhythm"]
        ton = data["tonal"]

        # 1. Timbre / Texture Features (~42 Dims)
        mfcc_mean = low["mfcc"]["mean"]
        if len(mfcc_mean) < 13: raise ValueError("MFCC mean length < 13")
        feature_vector.extend(mfcc_mean[1:13]) # 12

        mfcc_cov = low["mfcc"]["cov"]
        mfcc_variances = np.diag(mfcc_cov)
        if len(mfcc_variances) < 13: raise ValueError("MFCC cov diagonal length < 13")
        mfcc_std = np.sqrt(np.maximum(0, mfcc_variances[1:13]))
        feature_vector.extend(mfcc_std) # 12

        feature_vector.append(low["spectral_centroid"]["mean"]) # 1
        feature_vector.append(np.sqrt(max(0, low["spectral_centroid"]["var"]))) # 1
        feature_vector.append(low["spectral_flux"]["mean"]) # 1
        feature_vector.append(np.sqrt(max(0, low["spectral_flux"]["var"]))) # 1
        feature_vector.append(low["barkbands_flatness_db"]["mean"]) # 1
        feature_vector.append(low["spectral_entropy"]["mean"]) # 1
        feature_vector.append(low["zerocrossingrate"]["mean"]) # 1
        feature_vector.append(np.sqrt(max(0, low["zerocrossingrate"]["var"]))) # 1
        feature_vector.append(low["dissonance"]["mean"]) # 1
        feature_vector.append(np.sqrt(max(0, low["dissonance"]["var"]))) # 1
        feature_vector.append(low["pitch_salience"]["mean"]) # 1
        feature_vector.append(np.sqrt(max(0, low["pitch_salience"]["var"]))) # 1
        contrast_coeffs_mean = low["spectral_contrast_coeffs"]["mean"]
        if len(contrast_coeffs_mean) < 6: raise ValueError("Spectral Contrast Coeffs length < 6")
        feature_vector.extend(contrast_coeffs_mean) # 6

        # 2. Rhythmic / Groove Features (~13 Dims)
        feature_vector.append(rhy["bpm"]) # 1
        feature_vector.append(rhy["onset_rate"]) # 1
        feature_vector.append(rhy["danceability"]) # 1
        pulse_clarity = 0.0 # Default value
        peak_weight_value = rhy.get("bpm_histogram_first_peak_weight")
        if isinstance(peak_weight_value, dict):
            pulse_clarity = float(peak_weight_value.get("mean", 0.0))
        elif isinstance(peak_weight_value, (float, int)):
            pulse_clarity = float(peak_weight_value)

        feature_vector.append(pulse_clarity) # 1 dim
        band_ratio_mean = rhy["beats_loudness_band_ratio"]["mean"]
        if len(band_ratio_mean) < 6: raise ValueError("Band Ratio length < 6")
        feature_vector.extend(band_ratio_mean) # 6
        feature_vector.append(low["dynamic_complexity"]) # 1
        feature_vector.append(rhy["beats_count"]) # 1
        beats_pos = rhy["beats_position"]
        beat_interval_std = np.std(np.diff(beats_pos)) if len(beats_pos) > 1 else 0.0
        feature_vector.append(beat_interval_std) # 1

        # 3. Tonal / Harmonic Features (~39 Dims)
        hpcp_mean = ton["hpcp"]["mean"]
        if len(hpcp_mean) < 36: raise ValueError("HPCP Mean length < 36")
        feature_vector.extend(hpcp_mean) # 36
        feature_vector.append(ton["hpcp_entropy"]["mean"]) # 1
        feature_vector.append(ton.get("key_strength", ton.get("key_temperley", {}).get("strength", 0.0))) # 1
        key_scale_str = ton.get("key_scale", ton.get("key_temperley", {}).get("scale", "major"))    
        feature_vector.append(1.0 if key_scale_str == "major" else 0.0) # 1

        # --- DELTA FEATURES REMOVED ---

    except KeyError as e:
        print(f"{Fore.RED}Error extracting features from {os.path.basename(json_path)}: Missing Key -> {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"{Fore.RED}Error extracting features from {os.path.basename(json_path)}: Value Error -> {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Error extracting features from {os.path.basename(json_path)}: General Error -> {e}", file=sys.stderr)
        sys.exit(1)

     # Final check on extracted vector length
    if len(feature_vector) != EXPECTED_DIM:
        print(f"{Fore.RED}Error: Extracted feature vector length {len(feature_vector)} != expected {EXPECTED_DIM}", file=sys.stderr)
        sys.exit(1)

    # --- REVISED FINAL VALIDATION BLOCK (Type and Finite Check) ---
    try:
        for i, x in enumerate(feature_vector):
            # 1. Explicitly check the type FIRST
            if not isinstance(x, (int, float, np.number)):
                 print(f"\n{Fore.RED}DEBUG: Non-numeric type detected at index {i}", file=sys.stderr)
                 print(f"{Fore.RED}DEBUG: Value = {x}", file=sys.stderr)
                 print(f"{Fore.RED}DEBUG: Type = {type(x)}", file=sys.stderr)
                 # Raise error immediately upon finding wrong type
                 raise TypeError(f"Non-numeric value at index {i}")

            # 2. If it IS a number, check if it's finite (not NaN or Inf)
            if not np.isfinite(x):
                 print(f"\n{Fore.RED}DEBUG: Non-finite value (NaN/Inf) found at index {i}: value={x}", file=sys.stderr)
                 # Raise error immediately upon finding non-finite number
                 raise ValueError("Non-finite value (NaN or Inf) found in extracted feature vector.")

    except TypeError as e: # Catches error from the isinstance check OR isfinite if type was missed
         print(f"{Fore.RED}Exiting due to TYPE error during final vector check: {e}", file=sys.stderr)
         # The print statement inside the loop should have given details
         sys.exit(1)
    except ValueError as e: # Catches error from the isfinite check
         print(f"{Fore.RED}Exiting due to VALUE error (NaN/Inf) during final vector check: {e}", file=sys.stderr)
         # The print statement inside the loop should have given details
         sys.exit(1)
    # --- END REVISED BLOCK ---


    if verbose: print(f"{Fore.YELLOW}Extracted raw features ({len(feature_vector)} dims)", file=sys.stderr)
    return feature_vector # Return raw vector list only if all checks pass
# --- Qdrant Search Function ---
def perform_similarity_search(query_vector, top_k=TOP_K, collection=DEFAULT_COLLECTION_NAME, host=QDRANT_HOST, port=QDRANT_PORT):
    """Performs search using the provided query vector (expects a list)."""
    if not isinstance(query_vector, list):
        raise TypeError("query_vector must be a list")
    try:
        # Use http='auto' for automatic protocol selection (HTTP/HTTPS)
        # Increased timeout for potentially slower searches on large collections
        client = QdrantClient(host=host, port=port, timeout=20.0)
        # Verify connection and collection existence
        try:
            client.get_collection(collection_name=collection)
        except Exception as e:
            print(f"{Fore.RED}Error: Collection '{collection}' not found or connection failed @ {host}:{port}. {e}", file=sys.stderr)
            sys.exit(1)

        # Perform the search
        search_result = client.search(
            collection_name=collection,
            query_vector=query_vector, # Pass the list directly
            limit=top_k
        )
        return search_result
    except Exception as e:
        print(f"{Fore.RED}Error during Qdrant search: {e}", file=sys.stderr)
        sys.exit(1)
# ---------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Qdrant similarity client using Z-score normalization for ~{EXPECTED_DIM} features (unweighted).")
    parser.add_argument("query_json", help="Path to the query JSON file (features).")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME}).")
    parser.add_argument("--stats_file", default=DEFAULT_STATS_FILE, help=f"Path to the [mean, std_dev] stats file (default: {DEFAULT_STATS_FILE}).")
    parser.add_argument("--host", default=QDRANT_HOST, help=f"Qdrant host (default: read env or '{QDRANT_HOST}').")
    parser.add_argument("--port", type=int, default=QDRANT_PORT, help=f"Qdrant port (default: read env or {QDRANT_PORT}).")
    parser.add_argument("--json", action="store_true", help="Output results in raw JSON format.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help=f"Number of similar items (default: {TOP_K}).")
    parser.add_argument("--verbose", action="store_true", help="Print debug output to stderr.")
    # --- WEIGHT ARGUMENTS REMOVED ---
    args = parser.parse_args()

    if args.verbose: print(f"{Fore.YELLOW}--- Qdrant Similarity Client ({EXPECTED_DIM}-dim Z-Score Unweighted) ---", file=sys.stderr)
    verbose = args.verbose

    # 1) Extract raw features
    raw_vector = extract_features_from_json(args.query_json, verbose=verbose) # Returns list

    # 2) Load mean/std stats
    mean_std_stats = load_mean_std_stats(args.stats_file) # Returns list of lists

    # 3) Apply Z-score normalization
    zscore_vector = normalize_vector_zscore(raw_vector, mean_std_stats) # Returns list

    # --- WEIGHTING/SCALING STEP REMOVED ---
    # final_query_vector = zscore_vector (list)

    if verbose:
         raw_np = np.array(raw_vector); zscore_np = np.array(zscore_vector)
         np.set_printoptions(precision=4, suppress=True)
         print(f"{Fore.YELLOW}--- Vector Stages ---", file=sys.stderr)
         print(f"{Fore.CYAN}Raw Vector ({len(raw_vector)} dims excerpt): {Style.RESET_ALL}{raw_np[:5]}...{raw_np[-5:]}", file=sys.stderr)
         print(f"{Fore.CYAN}Z-Score Vector ({len(zscore_vector)} dims excerpt): {Style.RESET_ALL}{zscore_np[:5]}...{zscore_np[-5:]}", file=sys.stderr)
         zscore_norm = np.linalg.norm(zscore_np); print(f"{Fore.YELLOW}Z-Score Vector L2 Norm: {zscore_norm:.4f}", file=sys.stderr)
         print("-" * 20, file=sys.stderr)

    # 4) Perform similarity search using the UNWEIGHTED Z-SCORE vector
    results = perform_similarity_search(
        query_vector=zscore_vector, # Pass the Z-score list directly
        top_k=args.top_k,
        collection=args.collection,
        host=args.host,
        port=args.port
    )

    # 5) Output results
    if args.json:
        # Simple JSON output for unweighted results
        output_data = { "query_zscore_vector": zscore_vector, "results": [] }
        for r in results: output_data["results"].append({"id": r.id, "score": r.score, "payload": r.payload})
        print(json.dumps(output_data, indent=2))
    else:
        # Pretty print results
        print(f"\n{Fore.CYAN}{'='*80}"); print(f"{Fore.YELLOW}SIMILARITY SEARCH RESULTS ({EXPECTED_DIM}-dim Z-Score Unweighted)")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Collection: {args.collection}, Top K: {args.top_k}")
        print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}\n")
        if not results: print(f"{Fore.RED}No results found.")
        else:
            for i, res in enumerate(results):
                print(f"{Fore.GREEN}Match #{i+1} {Fore.CYAN}{'─'*50}")
                print(f"{Fore.YELLOW}ID: {Fore.WHITE}{res.id}")
                print(f"{Fore.YELLOW}Score (Cosine Sim): {Fore.WHITE}{res.score:.4f}")
                payload = res.payload or {} # Ensure payload exists
                artist = payload.get("artist", "N/A"); title = payload.get("title", "N/A"); mbid = payload.get("mbid", "N/A")
                print(f"{Fore.YELLOW}Artist: {Fore.WHITE}{artist}"); print(f"{Fore.YELLOW}Title: {Fore.WHITE}{title}"); print(f"{Fore.YELLOW}MBID: {Fore.WHITE}{mbid}")
                print("")