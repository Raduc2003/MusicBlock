#!/usr/bin/env python3
import os
import json
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models # Use models directly for PointStruct etc.
import argparse
import sys
import time # For potential retries or delays

# ------------------- Configuration -------------------

qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "music_similarity_zscore2" # New collection name suggested
STATS_FILE = "global_mean_std.json"   # File with [mean, std_dev] pairs
EXPECTED_DIM = 55     
FILE_LIMIT =1400000           # Expected dimension of feature vector
BATCH_SIZE = 1000                   # Number of points per upsert batch (REINTRODUCED)
MAX_WORKERS = 300                  # Number of parallel threads for file processing
# -----------------------------------------------------
# ------------------------------------------------------------------------------
# Create (or recreate) a collection for storing your vectors if it doesn't exist
# ------------------------------------------------------------------------------
collections = qdrant.get_collections().collections
collection_names = [collection.name for collection in collections]

if COLLECTION_NAME not in collection_names:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EXPECTED_DIM, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")

#Load the mean and std_dev stats from the JSON file
try:
    with open(STATS_FILE, "r") as f:
        global_mean_std_stats = json.load(f)
        if len(global_mean_std_stats) != EXPECTED_DIM:
            raise ValueError(f"Mean and std_dev stats length {len(global_mean_std_stats)} != expected {EXPECTED_DIM}")

        print(f"Loaded mean and std_dev stats from {STATS_FILE}.")
except FileNotFoundError:
    print(f"Error: File {STATS_FILE} not found. Please check the path.")
    sys.exit(1)


# --- Z-Score Function (remains the same) ---
def normalize_vector_zscore(vector, mean_std_stats):
    normalized = []
    if len(vector) != len(mean_std_stats):
        raise ValueError(f"Vector length {len(vector)} != stats length {len(mean_std_stats)}")
    for i, x in enumerate(vector):
        mean, std_dev = mean_std_stats[i]
        if std_dev < 1e-9:
            normalized_value = 0.0
        else:
            normalized_value = (x - mean) / std_dev
        normalized.append(normalized_value)
    return normalized

# --- Feature Extraction Function (remains the same) ---
def extract_features(json_path):
    """
    Extracts features and returns (vector, payload).
    
    The feature vector is constructed from:
      - 13 MFCC means (lowlevel.mfcc.mean)
      - 1 BPM, 1 onset_rate, 1 danceability (rhythm)
      - 1 key_strength, 1 chords_changes_rate, 1 hpcp_entropy.mean (tonal)
      - 36 HPCP means (tonal.hpcp.mean)
      
    Total dimensions: 13 + 3 + 3 + 36 = 55
    
    The payload includes:
      - key_key, key_scale, bpm, filename, artist, title
      - raw_vector: the unnormalized feature vector
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Low-level
    mfcc_mean = data["lowlevel"]["mfcc"]["mean"]  # 13 floats

    # Rhythmic
    bpm = data["rhythm"]["bpm"]
    onset_rate = data["rhythm"]["onset_rate"]
    danceability = data["rhythm"]["danceability"]

    # Tonal
    hpcp_mean = data["tonal"]["hpcp"]["mean"]     # 36 floats expected
    key_key = data["tonal"]["key_key"]
    key_scale = data["tonal"]["key_scale"]
    # Depending on your data, use either key_strength from "key_strength" or "key_temperley"
    key_strength = data["tonal"].get("key_strength", data["tonal"].get("key_temperley", {}).get("strength", 0))
    chords_changes_rate = data["tonal"]["chords_changes_rate"]
    hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]

    # Combine into a single feature vector
    feature_vector = (
        mfcc_mean +
        [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
        hpcp_mean
    )

    if len(feature_vector) != EXPECTED_DIM:
        raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected {EXPECTED_DIM}")

    # Retrieve artist and title if available; assume they are under "metadata.tags"
    artist = data["metadata"]["tags"]["artist"][0] if "artist" in data["metadata"]["tags"] else "Unknown"
    title = data["metadata"]["tags"]["title"][0] if "title" in data["metadata"]["tags"] else "Unknown"

    # Remove trailing "-[number].json" from filename
    term = re.sub(r"-\d+\.json$", "", os.path.basename(json_path))
    payload = {
        "key_key": key_key,
        "key_scale": key_scale,
        "bpm": bpm,
        "mbid": term,
        "artist": artist,
        "title": title,
        "raw_vector": feature_vector  # Store unnormalized vector for reference
    }
    return feature_vector, payload

def process_file(file_path: str, mean_std_stats: list) -> dict | None:
    raw_vector, payload = extract_features(file_path)
    if raw_vector is None: return None
    try:
        zscore_vector = normalize_vector_zscore(raw_vector, mean_std_stats)
        return {"vector": zscore_vector, "payload": payload}
    except Exception as e:
        print(f"Error normalizing vector from {os.path.basename(file_path)}: {e}")
        return None


# --- Main Indexing Function (Reintroduce Batching) ---
def index_directory_chunked(json_dir, qdrant_client, collection_name, mean_std_stats,
                            offset=0, limit=None):
    # File scanning logic (remains the same)
    file_paths = []
    print(f"Recursively scanning for JSON files in {json_dir}...")
    files_scanned = 0
    for root, _, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith(".json"):
                file_paths.append(os.path.join(root, filename))
                files_scanned +=1
                if files_scanned % 20000 == 0: print(f"  Scanned {files_scanned} files...")
    total_files_found = len(file_paths)
    print(f"Found {total_files_found} total JSON files.")
    file_paths.sort()

    # Apply Offset and Limit (remains the same)
    start_index = offset
    end_index = (offset + limit) if limit is not None else total_files_found
    files_to_process = file_paths[start_index:end_index]
    total_files_in_chunk = len(files_to_process)
    if total_files_in_chunk == 0: print(f"No files to process in chunk (offset={offset}, limit={limit})."); return
    print(f"Processing chunk: {total_files_in_chunk} files (from index {offset} up to {end_index}).")

    # Process files in parallel (remains the same logic, store results)
    points_data = []
    processed_count_in_chunk = 0
    skipped_count_in_chunk = 0
    filepath_to_index = {fp: i for i, fp in enumerate(files_to_process)}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_filepath = {executor.submit(process_file, fp, mean_std_stats): fp for fp in files_to_process}
        for future in future_to_filepath:
            processed_count_in_chunk += 1
            original_filepath = future_to_filepath[future]
            try:
                result = future.result()
                if result is not None:
                    original_index_in_chunk = filepath_to_index[original_filepath]
                    points_data.append({"data": result, "original_index": original_index_in_chunk})
                else: skipped_count_in_chunk += 1
            except Exception as e:
                 print(f"Error processing future for {os.path.basename(original_filepath)}: {e}")
                 skipped_count_in_chunk += 1
            if processed_count_in_chunk % 20000 == 0 or processed_count_in_chunk == total_files_in_chunk:
                  print(f"Worker progress: {processed_count_in_chunk}/{total_files_in_chunk} files processed...")

    print(f"Finished processing chunk. Valid vectors: {len(points_data)}, Skipped: {skipped_count_in_chunk}")
    if not points_data: print("No valid points generated. Nothing to upsert."); return

    # Assign IDs and Create PointStructs (remains the same)
    final_points = []
    for item in points_data:
        point_id = offset + item["original_index"]
        final_points.append(models.PointStruct(id=point_id, vector=item["data"]["vector"], payload=item["data"]["payload"]))

    if not final_points: print("Logical error: point data existed but final points list is empty."); return

    # --- Reintroduce Batch Upsert Loop ---
    total_points_to_upsert = len(final_points)
    num_batches = (total_points_to_upsert + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Upserting {total_points_to_upsert} points to Qdrant in {num_batches} batches (size {BATCH_SIZE})...")

    for i in range(0, total_points_to_upsert, BATCH_SIZE):
        batch_points = final_points[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Upserting batch {batch_num}/{num_batches} (Points {i + offset} to {i + offset + len(batch_points) - 1})...", end='', flush=True)
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=True # Wait for this batch operation to complete
            )
            print(" Done.")
        except Exception as e:
            print(f" Failed.\n    Error upserting batch {batch_num}: {e}")
            # Decide if you want to stop or continue on batch errors
            print("    Stopping upsert process due to error.")
            break # Stop processing further batches on error
        # Optional: Add a small delay between batches if needed
        # time.sleep(0.1)
    # --- End Batch Upsert Loop ---

    print(f"Upsert process completed for chunk (offset={offset}, limit={limit}).")

# --- Main Entry Point (remains the same) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Populate Qdrant with Z-score vectors (chunked).")
    parser.add_argument("json_folder", help="Root folder of feature JSON files.")
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"Qdrant collection name (default: {COLLECTION_NAME}).")
    parser.add_argument("--stats_file", default=STATS_FILE, help=f"Path to [mean, std_dev] stats JSON (default: {STATS_FILE}).")
    parser.add_argument("--offset", type=int, default=0, help="Starting file index (0-based) (default: 0).")
    parser.add_argument("--limit", type=int, default=None, help="Max number of files to process in this run (default: all).")
    parser.add_argument("--create", action="store_true", help="Create collection if it doesn't exist.")
    args = parser.parse_args()

  

    # Run Indexing
    index_directory_chunked( args.json_folder, qdrant, args.collection, global_mean_std_stats, args.offset, args.limit )
    print("\nPopulation script finished.")