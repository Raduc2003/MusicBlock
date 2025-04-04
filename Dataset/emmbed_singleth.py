import os
import json
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ------------------- Configuration -------------------
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "music_similarity4"
BATCH_SIZE = 1000      # Number of points per upsert batch
FILE_LIMIT = 500000    # Process up to 500k files
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
    global_min_max = json.load(f)  # List of 55 [min, max] pairs

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
      
    Total dimensions: 55
    
    The payload includes:
      - key_key, key_scale, bpm, filename, artist, title
      - raw_vector: the unnormalized feature vector
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    mfcc_mean = data["lowlevel"]["mfcc"]["mean"]  # 13 floats
    bpm = data["rhythm"]["bpm"]
    onset_rate = data["rhythm"]["onset_rate"]
    danceability = data["rhythm"]["danceability"]

    hpcp_mean = data["tonal"]["hpcp"]["mean"]     # 36 floats expected
    key_key = data["tonal"]["key_key"]
    key_scale = data["tonal"]["key_scale"]
    key_strength = data["tonal"].get("key_strength", data["tonal"].get("key_temperley", {}).get("strength", 0))
    chords_changes_rate = data["tonal"]["chords_changes_rate"]
    hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]

    feature_vector = (
        mfcc_mean +
        [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
        hpcp_mean
    )

    if len(feature_vector) != EXPECTED_DIM:
        raise ValueError(f"Feature vector length {len(feature_vector)} does not match expected {EXPECTED_DIM} for file {json_path}")

    artist = data["metadata"]["tags"]["artist"][0] if "artist" in data["metadata"]["tags"] else "Unknown"
    title = data["metadata"]["tags"]["title"][0] if "title" in data["metadata"]["tags"] else "Unknown"
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
# Process files sequentially
# ------------------------------------------------------------------------------
def process_files(json_dir, file_limit=None):
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
    for fp in file_paths:
        try:
            vector, payload = extract_features(fp)
            norm_vector = normalize_vector(vector, global_min_max)
            if len(norm_vector) != EXPECTED_DIM:
                print(f"Error: Normalized vector length {len(norm_vector)} does not match expected {EXPECTED_DIM} for file {fp}. Skipping.")
                continue
            # Optionally print debug info only for the first processed file
            if point_id == 0:
                print(f"\nFile: {fp}")
                print("Raw vector:")
                print(vector)
                print("Normalized vector:")
                print(norm_vector)
                print("-----------------------------------------")
            # Add the normalized vector and raw vector in payload
            payload["normalized_vector"] = norm_vector
            points.append({
                "id": point_id,
                "vector": norm_vector,  # Normalized vector used for similarity search
                "payload": payload      # Payload includes both raw_vector and normalized_vector
            })
            point_id += 1
        except Exception as e:
            print(f"Skipping {fp} due to error: {e}")
        
        if point_id % 1000 == 0 and point_id > 0:
            print(f"Processed {point_id}/{total_files} files...")
            
    print(f"Total valid points: {len(points)}")
    return points

# ------------------------------------------------------------------------------
# Upsert points in batches to Qdrant
# ------------------------------------------------------------------------------
def upsert_points(points):
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
    JSON_FOLDER = "acousticbrainz-lowlevel-json-20220623"
    points = process_files(JSON_FOLDER, file_limit=FILE_LIMIT)
    upsert_points(points)
