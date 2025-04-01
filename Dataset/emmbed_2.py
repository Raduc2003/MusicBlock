import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ------------------- Configuration -------------------
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "music_similarity2"
BATCH_SIZE = 1000      # Number of points per upsert batch
MAX_WORKERS = 12       # Number of parallel threads
FILE_LIMIT = 1400000    # Process up to 500k files
GLOBAL_MIN_MAX_FILE = "global_min_max.json"  # Precomputed global min/max values
EXPECTED_DIM = 55      # Expected dimension of feature vector
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

# -----------------------------------
# Load global min/max parameters from file
with open(GLOBAL_MIN_MAX_FILE, "r") as f:
    global_min_max = json.load(f)  # Should be a list of 55 [min, max] pairs

def normalize_vector(vector, min_max):
    """
    Normalize a single feature vector using the provided per-dimension min and max values.
    Each normalized value = (value - min) / (max - min).
    """
    normalized = []
    for i, x in enumerate(vector):
        mn, mx = min_max[i]
        range_val = mx - mn
        if range_val == 0:
            normalized.append(0.0)
        else:
            normalized.append((x - mn) / range_val)
    return normalized

# ------------------------------------------------------------------------------
# Helper function to extract features from a JSON file
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Wrapper function for threading
# ------------------------------------------------------------------------------
def process_file(file_path):
    try:
        vector, payload = extract_features(file_path)
        norm_vector = normalize_vector(vector, global_min_max)
        if len(norm_vector) != EXPECTED_DIM:
            print(f"Error: Normalized vector length {len(norm_vector)} does not match expected {EXPECTED_DIM} for file {file_path}.")
            return None
        # Print debug information for each file (or only for first few, as needed)
        print(f"\nFile: {file_path}")
        print("Raw vector:")
        print(vector)
        print("-----------------------------------------")
        print("Normalized vector:")
        print(norm_vector)
        print("-----------------------------------------")
        return norm_vector, payload
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return None

# ------------------------------------------------------------------------------
# Index files in parallel and upsert in batches (with normalization)
# ------------------------------------------------------------------------------
def index_directory(json_dir, file_limit=None):
    # Gather all JSON file paths recursively
    file_paths = []
    print(f"Scanning for JSON files in {json_dir}...")
    for root, dirs, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith(".json"):
                file_paths.append(os.path.join(root, filename))
                if file_limit and len(file_paths) >= file_limit:
                    break
        if file_limit and len(file_paths) >= file_limit:
            break

    total_files = len(file_paths)
    print(f"Found {total_files} files to process.")

    points = []
    point_id = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, fp) for fp in file_paths]
        for future in futures:
            result = future.result()
            if result is None:
                continue  # Skip files with errors
            norm_vector, payload = result
            points.append({
                "id": point_id,
                "vector": norm_vector,  # Normalized vector used for similarity search
                "payload": payload      # Payload includes raw_vector and normalized_vector info
            })
            point_id += 1
            if point_id % 1000 == 0:
                print(f"Processed {point_id}/{total_files} files...")

    print(f"Total valid points: {len(points)}")
    
    # Batch upsert to Qdrant
    print("Upserting batches to Qdrant...")
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        try:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            print(f"Upserted batch {i//BATCH_SIZE + 1}/{(len(points)-1)//BATCH_SIZE + 1} containing {len(batch)} points.")
        except Exception as e:
            print(f"Error upserting batch starting at index {i}: {e}")

    print(f"Indexed {len(points)} items into Qdrant collection '{COLLECTION_NAME}'.")

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    JSON_FOLDER = "/"
    index_directory(JSON_FOLDER, file_limit=FILE_LIMIT)
