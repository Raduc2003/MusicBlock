#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import tempfile
import shutil

# --- Defaults ---
DEFAULT_EXTRACTOR_CONFIG = "extraction_module/pr.yaml"
DEFAULT_PREPARE_SCRIPT = "prepare_extract.py" # Script inside extractor
DEFAULT_CLIENT_SCRIPT = "client_similarity.py"  # Script inside client
DEFAULT_TOP_K = 20
# ----------------

def run_full_pipeline(host_input_audio, extractor_config, top_k, json_output):
    """
    Orchestrates the extractor and client services using docker compose run.
    Runs on the HOST machine.
    """

    print(f"--- Starting Full Pipeline for: {host_input_audio} ---")

    # --- Define Paths ---
    # Host paths
    host_input_dir = os.path.dirname(host_input_audio)
    host_input_filename = os.path.basename(host_input_audio)
    base_filename = os.path.splitext(host_input_filename)[0]

    # Paths *inside* the extractor container
    extractor_input_audio_path = f"/data/input_audio/{host_input_filename}" # Corresponds to ./demoMusic mount
    extractor_output_json_dir = "/data/output_json" # Corresponds to shared_data mount
    extractor_config_path = f"/app/{os.path.basename(extractor_config)}" # Corresponds to pr.yaml mount

    # Path *inside* the client container
    client_input_json_path = f"/data/input_json/output_{base_filename}.json" # Corresponds to shared_data mount

    # --- Step 1: Run Extractor Service ---
    print("\n" + "="*15 + " Step 1: Running Extractor Service " + "="*15)
    cmd_extractor = [
        "docker","compose", "run", "--rm", # --rm cleans up container afterwards
        "extractor", # Service name in docker compose.yml
        "python3", DEFAULT_PREPARE_SCRIPT, # Command to run inside container
        extractor_input_audio_path,
        extractor_output_json_dir,
        "--extractor_config", extractor_config_path
        # Add --sr if needed
    ]
    print(f"Host executing: {' '.join(cmd_extractor)}")
    try:
        result_extractor = subprocess.run(cmd_extractor, check=True, capture_output=True, text=True)
        print("--- Extractor Output ---")
        print(result_extractor.stdout.strip())
        if result_extractor.stderr:
            print("--- Extractor Stderr ---")
            print(result_extractor.stderr.strip())
        print("--- Extractor Service Complete ---")
    except subprocess.CalledProcessError as e:
        print(f"\nError running extractor service (Return Code: {e.returncode}):")
        print(e.stderr.strip())
        sys.exit(1)
    except FileNotFoundError:
         print("Error: 'docker compose' command not found. Is it installed and in your PATH?")
         sys.exit(1)


    # --- Step 2: Run Client Service ---
    print("\n" + "="*15 + " Step 2: Running Client Service " + "="*15)
    cmd_client = [
        "docker","compose", "run", "--rm",
        "client", # Service name
        "python", DEFAULT_CLIENT_SCRIPT, # Command
        client_input_json_path,
        "--top_k", str(top_k)
    ]
    if json_output:
        cmd_client.append("--json")

    print(f"Host executing: {' '.join(cmd_client)}")
    try:
        # Let client print directly to host console
        result_client = subprocess.run(cmd_client, check=True, text=True)
        print("\n--- Client Service Complete ---")
    except subprocess.CalledProcessError as e:
        print(f"\nError running client service (Return Code: {e.returncode}). See output above.")
        sys.exit(1)
    except FileNotFoundError:
         print("Error: 'docker compose' command not found.")
         sys.exit(1)

    print("\n--- Full Pipeline Finished Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Extractor -> Client pipeline using Docker Compose."
    )
    parser.add_argument("input_audio", help="Path to the input audio file ON THE HOST.")
    parser.add_argument("--config", default=DEFAULT_EXTRACTOR_CONFIG, dest="extractor_config",
                        help=f"Path to the Essentia extractor config profile ON THE HOST (will be mounted). (default: {DEFAULT_EXTRACTOR_CONFIG})")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of similar items to retrieve (default: {DEFAULT_TOP_K}).")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output final similarity results in raw JSON format.")

    args = parser.parse_args()

    # Basic check if host files exist before starting
    if not os.path.isfile(args.input_audio):
        print(f"Error: Input audio file not found on host: {args.input_audio}")
        sys.exit(1)
    if not os.path.isfile(args.extractor_config):
        print(f"Error: Extractor config file not found on host: {args.extractor_config}")
        sys.exit(1)


    run_full_pipeline(
        host_input_audio=args.input_audio,
        extractor_config=args.extractor_config,
        top_k=args.top_k,
        json_output=args.json_output
    )