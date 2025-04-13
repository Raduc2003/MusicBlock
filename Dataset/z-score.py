#!/usr/bin/env python3
import os
import json
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
import argparse # Import argparse

# --- Constants ---
EXPECTED_DIM = 55
MAX_WORKERS = 300 # Adjust based on your system's cores/IO capability
# ---------------

def extract_features(json_path: str) -> list | None:
    """
    Extracts features from a JSON file and returns the 55-dimensional vector.
    Returns None if there's an error reading the file, extracting features,
    or if validation fails (dimension mismatch, non-finite values).

    Returns:
        list: The 55-dimensional feature vector or None if error.
    """
    try: # Add try-except around file reading
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        # print(f"Warning: File not found: {json_path}. Skipping.") # Can be too verbose
        return None
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {json_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: Error reading {json_path}: {e}. Skipping.")
        return None


    # --- Feature Extraction Logic (copied from previous script) ---
    try: # Add try-except around feature access
        # Low-level
        mfcc_mean = data["lowlevel"]["mfcc"]["mean"]
        # Rhythmic
        bpm = data["rhythm"]["bpm"]
        onset_rate = data["rhythm"]["onset_rate"]
        danceability = data["rhythm"]["danceability"]
        # Tonal
        hpcp_mean = data["tonal"]["hpcp"]["mean"]
        # Check if key_strength exists directly or nested under key_temperley
        key_strength = data["tonal"].get("key_strength", data["tonal"].get("key_temperley", {}).get("strength", 0))
        chords_changes_rate = data["tonal"]["chords_changes_rate"]
        hpcp_entropy_mean = data["tonal"]["hpcp_entropy"]["mean"]

        # Combine into a single feature vector
        feature_vector = (
            mfcc_mean +
            [bpm, onset_rate, danceability, key_strength, chords_changes_rate, hpcp_entropy_mean] +
            hpcp_mean
        )
    except KeyError as e:
         print(f"Warning: Missing key {e} in {json_path}. Skipping.")
         return None
    except Exception as e:
         print(f"Warning: Error extracting features from {json_path}: {e}. Skipping.")
         return None
    # --- End Feature Extraction ---

    # Basic validation
    if len(feature_vector) != EXPECTED_DIM:
        # Optional: print a warning or log this
        # print(f"Warning: Feature vector length {len(feature_vector)} != {EXPECTED_DIM} in {json_path}. Skipping.")
        return None
    # Check for non-finite values which would break calculations
    try: # Check for non-finite can fail if vector contains non-numeric types
        if not all(np.isfinite(x) for x in feature_vector):
            # Optional: print a warning or log this
            # print(f"Warning: Non-finite value found in feature vector for {json_path}. Skipping.")
            return None
    except TypeError:
        # Optional: print a warning or log this
        # print(f"Warning: Non-numeric value found in feature vector for {json_path}. Skipping.")
        return None


    return feature_vector


def process_file_for_stats(file_path: str) -> list | None:
    """Worker function for parallel processing."""
    try:
        return extract_features(file_path)
    except Exception as e:
        # Errors during feature extraction are now handled within extract_features
        # This catches unexpected errors during the process call itself
        print(f"Critical Error during worker process for {os.path.basename(file_path)}: {e}. Skipping.")
        return None


def calculate_stats_from_directory(json_dir: str, file_limit: int | None = None) -> list:
    """
    Scans a directory *recursively* for JSON files, extracts features in parallel,
    and returns a list of valid feature vectors.
    """
    file_paths = []
    print(f"Recursively scanning for JSON files in {json_dir}...")

    # --- Use os.walk for recursive search ---
    files_scanned = 0
    for root, dirs, files in os.walk(json_dir):
        # Optional: Skip hidden directories (like .git)
        # dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            if filename.endswith(".json"):
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)
                files_scanned +=1
                # Print progress during scan for very large directories
                if files_scanned % 20000 == 0: # Print less often
                     print(f"  Scanned {files_scanned} files...")
                if file_limit is not None and len(file_paths) >= file_limit:
                    print(f"Reached file limit ({file_limit}). Stopping scan.")
                    break # Stop processing files in current directory
        if file_limit is not None and len(file_paths) >= file_limit:
            break # Stop walking through directories
    # --- End recursive search ---

    total_files = len(file_paths)
    if total_files == 0:
        print("No JSON files found.")
        return []

    print(f"Found {total_files} JSON files to process.")

    vectors = []
    processed_count = 0
    skipped_count = 0

    # Use ThreadPoolExecutor to process files in parallel
    # Consider adjusting max_workers based on CPU cores and I/O speed
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        futures = {executor.submit(process_file_for_stats, fp): fp for fp in file_paths}

        # Collect results as they complete
        for future in futures:
             processed_count += 1 # Increment regardless of success
             try:
                 vector = future.result() # Get result (vector or None)
             except Exception as exc: # Catch potential errors from the future itself
                 print(f'File {futures[future]} generated an exception: {exc}')
                 vector = None

             if vector is not None:
                 vectors.append(vector)
             else:
                 skipped_count +=1 # Increment skipped count if result is None or future failed

             # Update progress less frequently for speed
             if processed_count % 10000 == 0 or processed_count == total_files:
                 print(f"Processed {processed_count}/{total_files} files... ({len(vectors)} valid vectors, {skipped_count} skipped)")

    print(f"Finished processing {total_files} files.")
    print(f"Total valid vectors extracted: {len(vectors)}")
    print(f"Files skipped due to errors or invalid data: {skipped_count}")
    return vectors


def calculate_global_mean_std(vectors: list) -> list | None:
    """
    Given a list of 55-dimensional feature vectors, compute the global mean
    and standard deviation for each dimension.

    Args:
        vectors (list): A list of lists/tuples, where each inner list/tuple
                        is a 55-dimensional feature vector.

    Returns:
        list: A list of [mean, std_dev] pairs for each dimension,
              or None if input is empty or invalid.
    """
    if not vectors:
        print("Error: No vectors provided to calculate statistics.")
        return None

    try:
        # Convert to NumPy array for efficient computation
        arr = np.array(vectors, dtype=np.float64) # Use float64 for precision

        # Ensure correct shape
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
             print(f"Error: Input data has unexpected shape {arr.shape}. Expected (n_samples, {EXPECTED_DIM}).")
             return None

        # Calculate mean along axis 0 (column-wise)
        mean_vals = np.mean(arr, axis=0)
        # Calculate standard deviation along axis 0 (column-wise)
        std_dev_vals = np.std(arr, axis=0)

        # Check for zero standard deviation (constant feature)
        # Use a small tolerance to account for floating point inaccuracies
        zero_std_tolerance = 1e-9
        zero_std_indices = np.where(std_dev_vals < zero_std_tolerance)[0]
        if len(zero_std_indices) > 0:
             print(f"Warning: Features at indices {list(zero_std_indices)} have near-zero standard deviation (< {zero_std_tolerance}).")
             # Consider setting std_dev to a small epsilon for these indices to avoid division by zero during normalization
             # std_dev_vals[zero_std_indices] = zero_std_tolerance

        # Combine into list of [mean, std_dev] pairs
        stats = [[mean_vals[i], std_dev_vals[i]] for i in range(arr.shape[1])]
        return stats

    except Exception as e:
        print(f"Error during NumPy calculations: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean and standard deviation for each feature dimension across a dataset of JSON files (searches subfolders)."
    )
    parser.add_argument("json_folder", help="Path to the root folder containing the feature JSON files.")
    parser.add_argument("output_file", default="global_mean_std.json", nargs='?', # Make optional
                        help="Path to save the calculated mean/std stats (default: global_mean_std.json).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on the number of files to process.")
    args = parser.parse_args()

    # Extract feature vectors from dataset recursively
    vectors = calculate_stats_from_directory(args.json_folder, file_limit=args.limit)

    if vectors: # Only proceed if we have valid vectors
        # Compute global mean and standard deviation
        global_stats = calculate_global_mean_std(vectors)

        if global_stats:
            print("\nGlobal mean and standard deviation for each dimension:")
            # Print only first few and last few for brevity if large
            print_limit = 5
            for i, (mean_val, std_dev_val) in enumerate(global_stats):
                if i < print_limit or i >= (EXPECTED_DIM - print_limit):
                     # Format for better alignment
                     print(f"Dim {i:>2}: mean = {mean_val:18.6f}, std_dev = {std_dev_val:18.6f}")
                elif i == print_limit:
                     print("...")

            # Save the stats to the specified output file
            try:
                with open(args.output_file, "w") as f:
                    # Convert numpy types to native Python types for JSON serialization
                    serializable_stats = [[float(mean), float(std)] for mean, std in global_stats]
                    json.dump(serializable_stats, f, indent=2) # Use indent=2 for readability
                print(f"\nGlobal mean/std values saved to {args.output_file}")
            except Exception as e:
                print(f"\nError saving stats to {args.output_file}: {e}")
    else:
         print("\nNo valid vectors collected, statistics calculation skipped.")