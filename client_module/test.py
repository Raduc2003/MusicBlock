#!/usr/bin/env python3
import json
import csv
import os
import subprocess
import shutil
import sys
from datetime import datetime

# --- Configuration (Paths relative to WORKDIR /app inside container) ---
# Since ./test is mounted to /app/test and WORKDIR is /app, these paths work:
EXPECTED_MATCHES_FILE = "test/test_config.json"
QUERY_JSON_FOLDER = "test/json_no_lnorm_beta2" # Base folder for query JSONs
RESULTS_CSV_FOLDER = "test/results/"
SESSION_FOLDER_BASE = "test/session"

# Container execution settings
# Change just this one line to switch between client implementations
CLIENT_SCRIPT = "94_client_similarity_z.py"  # Just the filename, not the full path
CLIENT_SERVICE_NAME = "client"

# Test Settings
TOP_K = 100
VARIANTS = ["f", "c", "30s"]
# --- End Configuration ---

# Create session and results folders inside the container's view of the mounted volume
# These paths are relative to the WORKDIR (/app)
session_folder = os.path.join(SESSION_FOLDER_BASE, datetime.now().strftime("%Y%m%d_%H%M%S"))
try:
    # Use os.makedirs to create intermediate directories if needed
    os.makedirs(session_folder, exist_ok=True)
except OSError as e:
    print(f"Error creating session folder {session_folder}: {e}")
    # Allow continuing, maybe permissions issue but test might still run partly
    # sys.exit(1) # Optional: Exit if session folder creation fails

try:
    os.makedirs(RESULTS_CSV_FOLDER, exist_ok=True)
except OSError as e:
     print(f"Error creating results folder {RESULTS_CSV_FOLDER}: {e}")
     # sys.exit(1) # Optional: Exit if results folder creation fails


def get_query_path(test_id, variant):
    """
    Given a test ID and variant, return the path to the corresponding
    features JSON, relative to the WORKDIR (/app).
    """
    # Paths are relative to /app now
    filename_underscore = f"output_{test_id}_{variant}.json"
    path_underscore = os.path.join(QUERY_JSON_FOLDER, filename_underscore)
    if os.path.exists(path_underscore):
        return path_underscore
    # Fallback: try the dash version
    filename_dash = f"output_{test_id}-{variant}.json"
    path_dash = os.path.join(QUERY_JSON_FOLDER, filename_dash)
    if os.path.exists(path_dash):
        return path_dash
    # If neither exists, return the underscore path
    return path_underscore


def main():
    # 1) Load the expected MBIDs from JSON (using path relative to /app)
    try:
        with open(EXPECTED_MATCHES_FILE, "r", encoding="utf-8") as f:
            expected_data = json.load(f)
    except FileNotFoundError:
         print(f"Error: Expected matches file not found at {EXPECTED_MATCHES_FILE}")
         print(f"  (Full path expected inside container: {os.path.abspath(EXPECTED_MATCHES_FILE)})")
         sys.exit(1)
    except Exception as e:
         print(f"Error reading expected matches file {EXPECTED_MATCHES_FILE}: {e}")
         sys.exit(1)

    # 2) Prepare CSV output (using path relative to /app)
    output_csv_filename = os.path.join(RESULTS_CSV_FOLDER, datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    fieldnames = [
        "test_id", "variant", "query_file_path", "expected_mbids", # Changed column name
        "found_in_top_1", "found_in_top_10", "found_in_top_20", "found_in_top_100",
        "actual_rank_found",
    ]

    try:
        with open(output_csv_filename, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            print(f"Starting tests. Results will be saved to: {output_csv_filename}")
            print(f"Session responses will be saved to: {session_folder}")
            print("-" * 30)

            # 3) Loop over each test ID in the expected data
            for test_id, mbid_list in expected_data.items():
                for variant in VARIANTS:
                    # Get path relative to /app
                    query_path = get_query_path(test_id, variant)

                    # If file doesn't exist (relative to /app), skip
                    if not os.path.isfile(query_path):
                        print(f"[WARNING] Query file not found for test {test_id} variant {variant}: {query_path}")
                        continue

                    print(f"[INFO] Running test {test_id} variant {variant}")
                    print(f"  Query Path (inside container): {query_path}")

                    # 4) Call client_similarity.py using the relative query path
                    # The script itself runs from /app, so this relative path works directly
                    cmd = [
                        # We are running THIS script, test.py, via docker compose run
                        # It will then call client_similarity.py DIRECTLY
                        # No need for another docker compose run here.
                        sys.executable, # The python interpreter inside the container
                        CLIENT_SCRIPT, # Relative path from /app works
                        query_path,             # Relative path from /app works
                        "--json",
                        "--top_k", str(TOP_K)
                        # We need to ensure client_similarity.py is also in /app
                    ]

                    try:
                        # Execute client_similarity.py directly within the same container
                        print(f"  Running command: {' '.join(cmd)}")
                        process = subprocess.run(cmd, capture_output=True, text=True, check=False)

                        if process.returncode != 0:
                            print(f"[ERROR] client_similarity.py failed for {query_path}")
                            print(f"  Return Code: {process.returncode}")
                            print(f"  Stderr: {process.stderr.strip()}")
                            print(f"  Stdout: {process.stdout.strip()}")
                            continue

                        # 5) Parse the JSON output
                        try:
                            result_json_str = process.stdout
                            json_start_index = result_json_str.find('{')
                            if json_start_index == -1:
                                raise json.JSONDecodeError("No JSON object found in output", result_json_str, 0)
                            result_json = json.loads(result_json_str[json_start_index:])

                        except json.JSONDecodeError as e:
                            print(f"[ERROR] Could not parse JSON from client script for {query_path}")
                            print(f"  Error: {e}")
                            print("  Raw output was:")
                            print(process.stdout)
                            continue

                    except FileNotFoundError:
                         # This would happen if client_similarity.py isn't found
                         print(f"[ERROR] Command '{cmd[1]}' not found. Is it copied/mounted correctly to /app?")
                         sys.exit(1)
                    except Exception as e:
                         print(f"[ERROR] Unexpected error running subprocess for {query_path}: {e}")
                         continue


                    # --- Processing results (same as before) ---
                    response_filename = f"response_{test_id}_{variant}.json"
                    response_path = os.path.join(session_folder, response_filename)
                    try:
                        with open(response_path, "w", encoding="utf-8") as resp_f:
                            json.dump(result_json, resp_f, indent=2)
                    except Exception as e:
                        print(f"[Warning] Could not save response JSON to {response_path}: {e}")


                    results = result_json.get("results", [])
                    found_in_top_1 = False
                    found_in_top_10 = False
                    found_in_top_20 = False
                    found_in_top_100 = False
                    actual_rank_found = None

                    for rank, item in enumerate(results, start=1):
                        mbid_in_payload = item.get("payload", {}).get("mbid")
                        if mbid_in_payload in mbid_list:
                            actual_rank_found = rank
                            break

                    if actual_rank_found is not None:
                         found_in_top_1 = (actual_rank_found == 1)
                         found_in_top_10 = (actual_rank_found <= 10)
                         found_in_top_20 = (actual_rank_found <= 20)
                         found_in_top_100 = (actual_rank_found <= TOP_K)

                    writer.writerow({
                        "test_id": test_id,
                        "variant": variant,
                        "query_file_path": query_path, # Store container-relative path
                        "expected_mbids": "|".join(mbid_list),
                        "found_in_top_1": found_in_top_1,
                        "found_in_top_10": found_in_top_10,
                        "found_in_top_20": found_in_top_20,
                        "found_in_top_100": found_in_top_100,
                        "actual_rank_found": actual_rank_found if actual_rank_found is not None else ""
                    })

                    print(f"  -> Done. Found rank: {actual_rank_found if actual_rank_found is not None else 'None'}")

    except Exception as e:
        print(f"\nAn error occurred during CSV writing: {e}")
        print("Output CSV might be incomplete.")

    print("\n" + "="*30)
    print("Testing finished.")
    print(f"Results saved to: {output_csv_filename}") # This path is inside the container's view
    print("Check the corresponding path on your host where './test/results' is mounted.")
    print("="*30)

if __name__ == "__main__":
    # No need to check Qdrant here, this script RUNS INSIDE the client container
    # which depends_on qdrant. Assume it's available or client_similarity will fail.
    main()