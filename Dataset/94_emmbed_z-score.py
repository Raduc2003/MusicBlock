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
COLLECTION_NAME = "zscore_94_1.4" # New collection name suggested
STATS_FILE = "global_mean_std_94FEATURES.json"   # File with [mean, std_dev] pairs
EXPECTED_DIM = 94
FILE_LIMIT = 1400000           # Expected dimension of feature vector
BATCH_SIZE = 1000                   # Number of points per upsert batch (REINTRODUCED)
MAX_WORKERS = 300                 # Number of parallel threads for file processing
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
    Extracts the REVISED ~94 features from a JSON file.
    Calculates MFCC std dev from covariance.
    Returns None if there's an error or validation fails.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        # print(f"ERROR reading {os.path.basename(json_path)}: {e}")
        return None

    try:
        feature_vector = []
        low = data["lowlevel"]
        rhy = data["rhythm"]
        ton = data["tonal"]

        # 1. Timbre / Texture Features (~42 Dims)
        mfcc_mean = low["mfcc"]["mean"]
        if len(mfcc_mean) < 13: raise ValueError("MFCC mean length")
        feature_vector.extend(mfcc_mean[1:13]) # 12 dims

        # Calculate MFCC Std Dev from Covariance Matrix Diagonal
        try:
            mfcc_cov = low["mfcc"]["cov"]
            mfcc_variances = np.diag(mfcc_cov)
            if len(mfcc_variances) < 13: raise ValueError("MFCC cov diagonal length")
            mfcc_std = np.sqrt(np.maximum(0, mfcc_variances[1:13]))
            feature_vector.extend(mfcc_std) # 12 dims
        except KeyError:
            print(f"ERROR: Missing 'cov' key for MFCC in {os.path.basename(json_path)}. Cannot calculate std dev.")
            return None

        # Spectral Centroid (Mean, StdDev) - Var assumed available based on sample
        feature_vector.append(low["spectral_centroid"]["mean"]) # 1
        try: feature_vector.append(np.sqrt(max(0, low["spectral_centroid"]["var"]))) # 1
        except KeyError: print(f"ERROR: Missing 'var' for spectral_centroid in {os.path.basename(json_path)}"); return None

        # Spectral Flux (Mean, StdDev) - Var assumed available based on sample
        feature_vector.append(low["spectral_flux"]["mean"]) # 1
        try: feature_vector.append(np.sqrt(max(0, low["spectral_flux"]["var"]))) # 1
        except KeyError: print(f"ERROR: Missing 'var' for spectral_flux in {os.path.basename(json_path)}"); return None

        # Spectral Flatness (Mean)
        try: feature_vector.append(low["barkbands_flatness_db"]["mean"]) # 1
        except KeyError: print(f"ERROR: Missing 'mean' for barkbands_flatness_db in {os.path.basename(json_path)}"); return None

        # Spectral Entropy (Mean)
        try: feature_vector.append(low["spectral_entropy"]["mean"]) # 1
        except KeyError: print(f"ERROR: Missing 'mean' for spectral_entropy in {os.path.basename(json_path)}"); return None

        # Zero-Crossing Rate (Mean, StdDev) - Var assumed available
        feature_vector.append(low["zerocrossingrate"]["mean"]) # 1
        try: feature_vector.append(np.sqrt(max(0, low["zerocrossingrate"]["var"]))) # 1
        except KeyError: print(f"ERROR: Missing 'var' for zerocrossingrate in {os.path.basename(json_path)}"); return None

        # Dissonance (Mean, StdDev) - Var assumed available
        feature_vector.append(low["dissonance"]["mean"]) # 1
        try: feature_vector.append(np.sqrt(max(0, low["dissonance"]["var"]))) # 1
        except KeyError: print(f"ERROR: Missing 'var' for dissonance in {os.path.basename(json_path)}"); return None

        # Pitch Salience (Mean, StdDev) - Var assumed available
        feature_vector.append(low["pitch_salience"]["mean"]) # 1
        try: feature_vector.append(np.sqrt(max(0, low["pitch_salience"]["var"]))) # 1
        except KeyError: print(f"ERROR: Missing 'var' for pitch_salience in {os.path.basename(json_path)}"); return None

        # Spectral Contrast Coeffs (Mean)
        contrast_coeffs_mean = low["spectral_contrast_coeffs"]["mean"]
        if len(contrast_coeffs_mean) < 6: raise ValueError("Spectral Contrast Coeffs length")
        feature_vector.extend(contrast_coeffs_mean) # 6 dims

        # 2. Rhythmic / Groove Features (~13 Dims) - No changes
        feature_vector.append(rhy["bpm"]) # 1
        feature_vector.append(rhy["onset_rate"]) # 1
        feature_vector.append(rhy["danceability"]) # 1
        feature_vector.append(rhy.get("bpm_histogram_first_peak_weight",{}).get("mean", 0.0)) # 1
        band_ratio_mean = rhy["beats_loudness_band_ratio"]["mean"]
        if len(band_ratio_mean) < 6: raise ValueError("Band Ratio length")
        feature_vector.extend(band_ratio_mean) # 6
        feature_vector.append(low["dynamic_complexity"]) # 1
        feature_vector.append(rhy["beats_count"]) # 1
        beats_pos = rhy["beats_position"]
        beat_interval_std = np.std(np.diff(beats_pos)) if len(beats_pos) > 1 else 0.0
        feature_vector.append(beat_interval_std) # 1

        # 3. Tonal / Harmonic Features (~39 Dims) - No changes
        hpcp_mean = ton["hpcp"]["mean"]
        if len(hpcp_mean) < 36: raise ValueError("HPCP Mean length")
        feature_vector.extend(hpcp_mean) # 36
        feature_vector.append(ton["hpcp_entropy"]["mean"]) # 1
        feature_vector.append(ton.get("key_strength", ton.get("key_temperley", {}).get("strength", 0.0))) # 1
        key_scale_str = ton["key_scale"]
        feature_vector.append(1.0 if key_scale_str == "major" else 0.0) # 1


    except KeyError as e:
         print(f"ERROR extracting features from {os.path.basename(json_path)}: Missing Key -> {e}")
         return None
    except ValueError as e:
         print(f"ERROR extracting features from {os.path.basename(json_path)}: Value Error -> {e}")
         return None
    except Exception as e:
         print(f"ERROR extracting features from {os.path.basename(json_path)}: General Error -> {e}")
         return None

    if len(feature_vector) != EXPECTED_DIM:
        raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected {EXPECTED_DIM}")

    # Retrieve artist and title if available; assume they are under "metadata.tags"
    artist = data["metadata"]["tags"]["artist"][0] if "artist" in data["metadata"]["tags"] else "Unknown"
    title = data["metadata"]["tags"]["title"][0] if "title" in data["metadata"]["tags"] else "Unknown"
    
    # Remove trailing "-[number].json" from filename
    term = re.sub(r"-\d+\.json$", "", os.path.basename(json_path))
    payload = {
        "key_key": ton["key_key"],
        "key_scale": key_scale_str,
        "bpm": rhy["bpm"],
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
    parser.add_argument("--limit", type=int, default=FILE_LIMIT, help="Max number of files to process in this run (default: all).")
    parser.add_argument("--create", action="store_true", help="Create collection if it doesn't exist.")
    args = parser.parse_args()

  

    # Run Indexing
    index_directory_chunked( args.json_folder, qdrant, args.collection, global_mean_std_stats, args.offset, args.limit )
    print("\nPopulation script finished.")