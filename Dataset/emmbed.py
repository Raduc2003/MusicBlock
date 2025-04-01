import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install via: pip install qdrant-client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import re

# ------------------- Configuration -------------------
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "music_similarity5"
BATCH_SIZE = 1000    # Number of points per upsert batch
MAX_WORKERS = 12      # Number of parallel threads
FILE_LIMIT = 1000  # Process 100k files
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
        vectors_config=VectorParams(size=55, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")

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
    
    The payload now includes:
      - key_key
      - key_scale
      - bpm
      - filename
      - artist
      - title
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
    key_strength = data["tonal"]["key_strength"]
    chords_changes_rate = data["tonal"]["chords_changes_rate"]
    hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]

    # Combine into a single feature vector - strings (key_key, key_scale) excluded from vector
    feature_vector = (
        mfcc_mean +
        [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
        hpcp_mean
    )

    # Retrieve artist and title if available; assume they are under "metadata"
    artist =data["metadata"]["tags"]["artist"][0] if "artist" in data["metadata"]["tags"] else "Unknown"
    title = data["metadata"]["tags"]["title"][0] if "title" in data["metadata"]["tags"] else "Unknown"

    # Store additional metadata as payload
    # termination of filename -[any number].json
    term = re.sub(r"-\d+\.json$", "", os.path.basename(json_path))
    payload = {
        "key_key": key_key,
        "key_scale": key_scale,
        "bpm": bpm,
        "mbid": term,
        "artist": artist,
        "title": title
    }
    return feature_vector, payload

# ------------------------------------------------------------------------------
# Wrapper function for threading
# ------------------------------------------------------------------------------
def process_file(file_path):
    try:
        return extract_features(file_path)
    except KeyError as e:
        print(f"Skipping {file_path} due to missing key: {e}")
        return None

# ------------------------------------------------------------------------------
# Index files in parallel and upsert in batches (without normalization)
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

    print(f"Found {len(file_paths)} files to process.")

    points = []
    point_id = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_file, fp): fp for fp in file_paths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result is None:
                continue  # Skip files with errors
            vector, payload = result
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": payload
            })
            point_id += 1
            if point_id % 1000 == 0:
                print(f"Processed {point_id} files...")

    print(f"Total valid points: {len(points)}")
    
    # Batch upsert to avoid timeouts
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
    # Process files with increased limit
    JSON_FOLDER = "acousticbrainz-lowlevel-json-20220623"
    index_directory(JSON_FOLDER, file_limit=FILE_LIMIT)
