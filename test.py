#!/usr/bin/env python3
import json
import csv
import os
import subprocess
import shutil
from datetime import datetime

EXPECTED_MATCHES_FILE = "/home/raduc/Documents/Code/Licentzio/test/test_config.json"  # Expected matches JSON file
QUERY_JSON_FOLDER = "test/json"                  # Where your output_X_variant.json files live
RESULTS_CSV = "test/results/"
CLIENT_SCRIPT = "client_similarity.py"
TOP_K = 100
VARIANTS = ["f", "c", "30s"]  # Variants to test for each test id

# Create a new session folder using a timestamp to store DB responses
session_folder = os.path.join("test", "session", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(session_folder, exist_ok=True)

def get_query_path(test_id, variant):
    """
    Given a test ID and variant, return the path to the corresponding features JSON.
    It first tries the underscore version (e.g. output_10_f.json) and if that doesn't exist,
    it falls back to the dash version (e.g. output_10-f.json).
    """
    filename_underscore = f"output_{test_id}_{variant}.json"
    path_underscore = os.path.join(QUERY_JSON_FOLDER, filename_underscore)
    if os.path.exists(path_underscore):
        return path_underscore
    # Fallback: try the dash version
    filename_dash = f"output_{test_id}-{variant}.json"
    path_dash = os.path.join(QUERY_JSON_FOLDER, filename_dash)
    if os.path.exists(path_dash):
        return path_dash
    # If neither exists, return the underscore path (which will likely fail later)
    return path_underscore

def main():
    # 1) Load the expected MBIDs from JSON
    with open(EXPECTED_MATCHES_FILE, "r", encoding="utf-8") as f:
        expected_data = json.load(f)
    
    # 2) Prepare CSV output (adding found_in_top_100 column)
    fieldnames = [
        "test_id",
        "variant",
        "query_file",
        "expected_mbids",
        "found_in_top_1",
        "found_in_top_10",
        "found_in_top_20",
        "found_in_top_100",
        "actual_rank_found",  # Rank of the first expected MBID match, if any
    ]
    with open(RESULTS_CSV + datetime.now().strftime("%Y%m%d_%H%M%S")+".csv", "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 3) Loop over each test ID in the expected data
        for test_id, mbid_list in expected_data.items():
            for variant in VARIANTS:
                query_file = get_query_path(test_id, variant)
                
                # If there's no corresponding JSON file, skip this variant
                if not os.path.isfile(query_file):
                    print(f"[WARNING] Query file not found for test {test_id} variant {variant}: {query_file}")
                    continue
                
                # 4) Call client_similarity.py with --json
                cmd = [
                    "python", CLIENT_SCRIPT,
                    query_file,
                    "--json",
                    "--top_k", str(TOP_K)
                ]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    print(f"[ERROR] client_similarity.py failed for {query_file}\n{process.stderr}")
                    continue
                
                # 5) Parse the JSON output (the DB response)
                try:
                    result_json = json.loads(process.stdout)
                except json.JSONDecodeError:
                    print(f"[ERROR] Could not parse JSON from client script for {query_file}")
                    print("Raw output was:")
                    print(process.stdout)
                    continue
                
                # Save the DB response JSON to the session folder
                response_filename = f"response_{test_id}_{variant}.json"
                response_path = os.path.join(session_folder, response_filename)
                with open(response_path, "w", encoding="utf-8") as resp_f:
                    json.dump(result_json, resp_f, indent=2)
                
                results = result_json.get("results", [])
                
                # 6) Check if any expected MBIDs appear in the results (top-1, top-10, top-20, top-100)
                found_in_top_1 = False
                found_in_top_10 = False
                found_in_top_20 = False
                found_in_top_100 = False
                actual_rank_found = None
                
                # Find the best (lowest) rank for any expected MBID
                for rank, item in enumerate(results, start=1):
                    mbid_in_payload = item["payload"].get("mbid")
                    if mbid_in_payload in mbid_list:
                        actual_rank_found = rank
                        break  # Stop on first match
                
                if actual_rank_found == 1:
                    found_in_top_1 = True
                if actual_rank_found and actual_rank_found <= 10:
                    found_in_top_10 = True
                if actual_rank_found and actual_rank_found <= 20:
                    found_in_top_20 = True
                if actual_rank_found and actual_rank_found <= 100:
                    found_in_top_100 = True
                
                # 7) Write row to CSV
                writer.writerow({
                    "test_id": test_id,
                    "variant": variant,
                    "query_file": query_file,
                    "expected_mbids": "|".join(mbid_list),
                    "found_in_top_1": found_in_top_1,
                    "found_in_top_10": found_in_top_10,
                    "found_in_top_20": found_in_top_20,
                    "found_in_top_100": found_in_top_100,
                    "actual_rank_found": actual_rank_found if actual_rank_found else ""
                })
                
                print(f"[INFO] Test {test_id} variant {variant} done. Found rank: {actual_rank_found}")

if __name__ == "__main__":
    main()
