#!/usr/bin/env python3
import os
import json
from concurrent.futures import ThreadPoolExecutor
import re
import numpy as np
import argparse
import sys

# --- Constants ---
# Timbre: 12(MFCCm)+12(MFCCs)+2(Cent_m_s)+2(Flux_m_s)+1(Flat_m)+1(Entr_m)+2(ZCR_m_s)+2(Diss_m_s)+2(PSal_m_s)+6(Contr_m) = 42
# Rhythm: 1(BPM)+1(OR)+1(Dance)+1(Pulse)+6(BandRatio)+1(DynCom)+1(BCnt)+1(BIntStd) = 13 dims
# Tonal: 36(HPCPm)+1(HPCPent)+1(KeyStr)+1(KeyScl) = 39 dims

# Total= 42+13+39 = 94
EXPECTED_DIM = 94 
MAX_WORKERS = 30 # Adjust based on your system
# ---------------

def extract_features(json_path: str) -> list | None:
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

    # --- Final Validation ---
    current_len = len(feature_vector)
    if current_len != EXPECTED_DIM:
        print(f"ERROR in {os.path.basename(json_path)}: Final vector length {current_len} != expected {EXPECTED_DIM}.")
        return None
    try:
        if not all(np.isfinite(x) for x in feature_vector):
            print(f"ERROR in {os.path.basename(json_path)}: Non-finite value found in final vector.")
            return None
    except TypeError as e:
        print(f"ERROR in {os.path.basename(json_path)}: Non-numeric value check failed -> {e}")
        return None

    return feature_vector


def process_file_for_stats(file_path: str) -> list | None:
    """Worker function for parallel processing."""
    try:
        return extract_features(file_path)
    except Exception as e:
        print(f"Critical Error during worker process for {os.path.basename(file_path)}: {e}. Skipping.")
        return None

# --- calculate_stats_from_directory remains the same ---
def calculate_stats_from_directory(json_dir: str, file_limit: int | None = None) -> list:
    """
    Scans a directory *recursively* for JSON files, extracts features in parallel,
    and returns a list of valid feature vectors.
    """
    file_paths = []
    print(f"Recursively scanning for JSON files in {json_dir}...")
    files_scanned = 0
    for root, _, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith(".json"):
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)
                files_scanned +=1
                if files_scanned % 20000 == 0:
                     print(f"  Scanned {files_scanned} files...")
                if file_limit is not None and len(file_paths) >= file_limit:
                    print(f"Reached file limit ({file_limit}). Stopping scan.")
                    break
        if file_limit is not None and len(file_paths) >= file_limit:
            break

    total_files = len(file_paths)
    if total_files == 0: print("No JSON files found."); return []
    print(f"Found {total_files} JSON files to process.")

    vectors = []
    processed_count = 0
    skipped_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_for_stats, fp): fp for fp in file_paths}
        for future in futures:
             processed_count += 1
             try:
                 vector = future.result()
             except Exception as exc:
                 print(f'File {futures[future]} generated an exception: {exc}')
                 vector = None
             if vector is not None:
                 vectors.append(vector)
             else:
                 skipped_count +=1
             if processed_count % 10000 == 0 or processed_count == total_files:
                 print(f"Processed {processed_count}/{total_files} files... ({len(vectors)} valid vectors, {skipped_count} skipped)")

    print(f"Finished processing {total_files} files.")
    print(f"Total valid vectors extracted: {len(vectors)}")
    print(f"Files skipped due to errors or invalid data: {skipped_count}")
    return vectors

# --- calculate_global_mean_std remains the same ---
def calculate_global_mean_std(vectors: list) -> list | None:
    """
    Given a list of feature vectors (NEW DIM), compute the global mean
    and standard deviation for each dimension.
    """
    if not vectors:
        print("Error: No vectors provided to calculate statistics.")
        return None
    try:
        arr = np.array(vectors, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
             print(f"Error: Input data shape {arr.shape} does not match EXPECTED_DIM={EXPECTED_DIM}.")
             return None

        mean_vals = np.mean(arr, axis=0)
        std_dev_vals = np.std(arr, axis=0)

        zero_std_tolerance = 1e-9
        zero_std_indices = np.where(std_dev_vals < zero_std_tolerance)[0]
        if len(zero_std_indices) > 0:
             print(f"Warning: Features at indices {list(zero_std_indices)} have near-zero std dev (< {zero_std_tolerance}).")

        stats = [[mean_vals[i], std_dev_vals[i]] for i in range(arr.shape[1])]
        return stats

    except Exception as e:
        print(f"Error during NumPy calculations: {e}")
        return None

# --- main block updated ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Calculate mean/std dev for the FINAL ~{EXPECTED_DIM}-dim feature set (MFCC cov->std, no deltas) across JSON files (recursive)."
    )
    parser.add_argument("json_folder", help="Root folder containing feature JSON files.")
    parser.add_argument("output_file", default=f"global_mean_std_{EXPECTED_DIM}FEATURES.json", nargs='?', # Updated default name
                        help=f"Path to save calculated mean/std stats (default: global_mean_std_{EXPECTED_DIM}FEATURES.json).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on the number of files to process.")
    args = parser.parse_args()

    # Turn on error printing during extraction for debugging
    print("Processing files (error messages during extraction will be shown)...")
    vectors = calculate_stats_from_directory(args.json_folder, file_limit=args.limit)

    if vectors:
        global_stats = calculate_global_mean_std(vectors)

        if global_stats:
            print(f"\nGlobal mean and standard deviation for {EXPECTED_DIM} dimensions:")
            print_limit = 5
            for i, (mean_val, std_dev_val) in enumerate(global_stats):
                if i < print_limit or i >= (EXPECTED_DIM - print_limit):
                     print(f"Dim {i:>3}: mean = {mean_val:18.6f}, std_dev = {std_dev_val:18.6f}")
                elif i == print_limit:
                     print("...")

            try:
                with open(args.output_file, "w") as f:
                    serializable_stats = [[float(mean), float(std)] for mean, std in global_stats]
                    json.dump(serializable_stats, f, indent=2)
                print(f"\nGlobal mean/std values for {EXPECTED_DIM} features saved to {args.output_file}")
            except Exception as e:
                print(f"\nError saving stats to {args.output_file}: {e}")
    else:
         print("\nNo valid vectors collected, statistics calculation skipped.")