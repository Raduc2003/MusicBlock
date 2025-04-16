#!/usr/bin/env python3
import os
import json
import sys
import numpy as np
from qdrant_client import QdrantClient
from colorama import init, Fore, Style
import argparse

# Initialize colorama
init(autoreset=True)

# ------------------- Configuration -------------------
COLLECTION_NAME = "music_similarity2"
TOP_K = 20  # Number of similar items to retrieve
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
GLOBAL_MIN_MAX_FILE = "global_min_max.json"  # File with 55 [min, max] pairs
# -----------------------------------------------------

def load_global_min_max(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def normalize_vector(vector, min_max):
    normalized = []
    for i, x in enumerate(vector):
        mn, mx = min_max[i]
        range_val = mx - mn
        if range_val == 0:
            normalized.append(0.0)
        else:
            normalized.append((x - mn) / range_val)
    return normalized

def extract_features_from_json(json_path, verbose=True):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Low-level features
    mfcc_mean = data["lowlevel"]["mfcc"]["mean"]  # 13 floats
    
    # Rhythmic features
    bpm = data["rhythm"]["bpm"]
    onset_rate = data["rhythm"]["onset_rate"]
    danceability = data["rhythm"]["danceability"]
    
    # Tonal features
    #key_strength = data["tonal"]["key_strength"] or data["tonal"]["key_temperley"]["strength"]
    # The key_strength is extracted from either "key_strength" or "key_temperley" based on availability
    key_strength = data["tonal"].get("key_strength", data["tonal"].get("key_temperley", {}).get("strength", 0))

    chords_changes_rate = data["tonal"]["chords_changes_rate"]
    hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]
    hpcp_mean = data["tonal"]["hpcp"]["mean"]     # 36 floats
    
    # Combine into a single feature vector (55 dims)
    feature_vector = (
        mfcc_mean +
        [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
        hpcp_mean
    )
    
    if verbose:
        # Print debug info to stderr (so stdout remains clean for JSON)
        print(f"{Fore.YELLOW}Extracted raw feature vector (55 dims):\n{feature_vector}\n", file=sys.stderr)
    return feature_vector

def perform_similarity_search(query_vector, top_k=TOP_K):
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant similarity client")
    parser.add_argument("query_json", help="Path to the query JSON file (features).")
    parser.add_argument("--json", action="store_true", 
                        help="Output results in raw JSON format.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of similar items to retrieve.")
    parser.add_argument("--verbose", action="store_true", help="Print debug output.")
    args = parser.parse_args()
    
    # If --json is set, we disable verbose to avoid polluting stdout with debug info.
    verbose = args.verbose and not args.json

    # 1) Extract raw features (suppress debug output if --json is used)
    raw_vector = extract_features_from_json(args.query_json, verbose=verbose)
    
    # 2) Load global min/max and normalize
    global_min_max = load_global_min_max(GLOBAL_MIN_MAX_FILE)
    normalized_vector = normalize_vector(raw_vector, global_min_max)
    
    # 3) Perform similarity search
    results = perform_similarity_search(normalized_vector, top_k=args.top_k)
    
    if args.json:
        output_data = {
            "query_vector": raw_vector,
            "normalized_query_vector": normalized_vector,
            "results": []
        }
        for r in results:
            output_data["results"].append({
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            })
        # Only JSON output on stdout
        print(json.dumps(output_data, indent=2))
    else:
        # Color-coded output if not in JSON mode
        from colorama import Style
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}QUERY VECTOR")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        print(f"{Fore.YELLOW}Raw vector: {Fore.WHITE}{raw_vector}")
        print(f"{Fore.YELLOW}Normalized vector: {Fore.WHITE}{normalized_vector}\n")
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.YELLOW}SEARCH RESULTS")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        print(f"{Fore.YELLOW}Top {args.top_k} matches:\n")
        print(f"{Fore.YELLOW}SIMILARITY QUERY RESULTS")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        for i, res in enumerate(results):
            print(f"{Fore.GREEN}Match #{i+1} {Fore.CYAN}{'─'*50}")
            print(f"{Fore.YELLOW}ID: {Fore.WHITE}{res.id}")
            print(f"{Fore.YELLOW}Score: {Fore.WHITE}{res.score:.4f}")
            #print artist name and title
            if "artist" in res.payload and "title" in res.payload:
                print(f"{Fore.YELLOW}Artist: {Fore.WHITE}{res.payload['artist']}")
                print(f"{Fore.YELLOW}Title: {Fore.WHITE}{res.payload['title']}")
            print(f"{Fore.YELLOW}Payload: \n{Fore.WHITE}{res.payload}\n")
