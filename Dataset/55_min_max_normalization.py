import os
import json
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np

def extract_features(json_path):
    """
    Extracts features and returns the 55-dimensional vector.
    
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
      - filename (modified)
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

    # Combine into a single feature vector
    feature_vector = (
        mfcc_mean +
        [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
        hpcp_mean
    )

    # Retrieve artist and title if available; assume they are under "metadata.tags"
    artist = data["metadata"]["tags"]["artist"][0] if "artist" in data["metadata"]["tags"] else "Unknown"
    title = data["metadata"]["tags"]["title"][0] if "title" in data["metadata"]["tags"] else "Unknown"

    # Modify filename: remove trailing -[number].json if present
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

def process_file(file_path):
    try:
        return extract_features(file_path)
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        return None

def index_directory(json_dir, file_limit=None):
    # Gather all JSON file paths recursively
    file_paths = []
    print(f"Scanning for JSON files in {json_dir}...")
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".json"):
                file_paths.append(os.path.join(root, file))
                if file_limit is not None and len(file_paths) >= file_limit:
                    break
        if file_limit is not None and len(file_paths) >= file_limit:
            break

    total_files = len(file_paths)
    print(f"Found {total_files} JSON files to process.")

    vectors = []
    payloads = []
    processed_count = 0

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_file, fp) for fp in file_paths]
        for future in futures:
            result = future.result()
            processed_count += 1
            if result is not None:
                vector, payload = result
                vectors.append(vector)
                payloads.append(payload)
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{total_files} files...")

    print(f"Total valid points extracted: {len(vectors)}")
    return vectors, payloads

def find_global_min_max(vectors):
    """
    Given a list of 55-dimensional feature vectors, compute the global min and max for each dimension.
    Returns a list of [min, max] pairs for each dimension.
    """
    arr = np.array(vectors)  # shape: (n_samples, 55)
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    return [[min_vals[i], max_vals[i]] for i in range(arr.shape[1])]

if __name__ == "__main__":
    JSON_FOLDER = "./"
    FILE_LIMIT = None # Adjust as needed (or set to None for all files)

    # Extract feature vectors from dataset
    vectors, payloads = index_directory(JSON_FOLDER, file_limit=FILE_LIMIT)
    
    # Compute global min and max for each of the 55 dimensions
    global_min_max = find_global_min_max(vectors)
    print("Global min and max for each dimension:")
    for i, (min_val, max_val) in enumerate(global_min_max):
        print(f"Dimension {i}: min = {min_val}, max = {max_val}")

    # Optionally, save the min/max values for future use
    with open("global_min_max.json", "w") as f:
        json.dump(global_min_max, f, indent=4)
    print("Global min/max values saved to global_min_max.json")
