#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import tempfile
import essentia.standard as es
import numpy as np # Required for checking silence

def prepare_and_extract(input_audio, output_folder, temp_dir, sr=44100,
                         extractor_config="pr.yaml"):
    """
    Loads audio, resamples to SR, converts to mono, saves temporarily,
    then runs Essentia extractor and saves JSON to output_folder.
    
    Will try 'essentia_streaming_extractor_music' first, and if not found,
    fall back to 'streaming_extractor_music'.

    Parameters:
      input_audio (str): Path to the input audio file.
      output_folder (str): Directory to save the output JSON features.
      temp_dir (str): Directory for temporary processed WAV file.
      sr (int): Target sample rate.
      extractor_config (str): Path to the extractor configuration profile (e.g., pr.yaml).
    """
    print(f"--- Processing: {input_audio} ---")

    # --- 1. Prepare Audio (Load, Resample, Mono) ---
    temp_wav_path = None # Initialize in case of early exit
    try:
        print(f"Step 1: Loading, resampling to {sr} Hz, converting to mono...")
        loader = es.MonoLoader(filename=input_audio, sampleRate=sr)
        audio = loader()

        # Handle potential silent file (extractor might fail)
        if np.max(np.abs(audio)) == 0.0:
             print(f"Warning: Input file {input_audio} appears to be silent. Skipping extraction.")
             return # Exit processing for this file

        # Create temporary file path
        # Use a unique name to avoid collisions if running in parallel
        fd, temp_wav_path = tempfile.mkstemp(suffix=".wav", prefix="prep_", dir=temp_dir)
        os.close(fd) # Close the file descriptor, we just need the path

        print(f"Step 2: Saving temporary processed WAV to {temp_wav_path}...")
        writer = es.MonoWriter(filename=temp_wav_path, sampleRate=sr, format='wav')
        writer(audio)
        print("Temporary WAV saved successfully.")

    except Exception as e:
        print(f"Error during audio preparation for {input_audio}: {e}")
        # Clean up temporary file if it was created
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                print(f"Cleaned up temporary file: {temp_wav_path}")
            except OSError as rm_err:
                print(f"Error removing temporary file {temp_wav_path}: {rm_err}")
        return # Stop processing this file

    # --- 2. Extract Features ---
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Construct output JSON filename
        base_filename = os.path.splitext(os.path.basename(input_audio))[0]
        output_json_path = os.path.join(output_folder, f"output_{base_filename}.json")

        print(f"Step 3: Running Essentia extractor...")
        print(f"  Input: {temp_wav_path}")
        print(f"  Output: {output_json_path}")
        print(f"  Config: {extractor_config}")

        # Check if config file exists
        if not os.path.isfile(extractor_config):
            #  print the full path to the config file
            print(f"Extractor config file not found: {os.path.abspath(extractor_config)}")
             # Manual cleanup needed here before returning
            raise FileNotFoundError(f"Extractor config not found: {extractor_config}")

        # Define the two extractor commands to try
        extractor_commands = [
            "essentia_streaming_extractor_music",
            "streaming_extractor_music"
        ]
        
        extraction_success = False
        
        for i, extractor_path in enumerate(extractor_commands):
            print(f"Trying extractor command {i+1}/{len(extractor_commands)}: {extractor_path}")
            
            extractor_cmd = [
                extractor_path,
                temp_wav_path,
                output_json_path,
                extractor_config
            ]

            print(f"Executing command: {' '.join(extractor_cmd)}")
            
            try:
                # Use subprocess.run for better error capture
                result = subprocess.run(extractor_cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    print(f"Feature extraction successful using '{extractor_path}'.")
                    extraction_success = True
                    break  # Exit the loop if successful
                else:
                    print(f"Command '{extractor_path}' failed with return code {result.returncode}")
                    print(f"  Stderr: {result.stderr}")
                    print(f"  Stdout: {result.stdout}")
            except FileNotFoundError:
                print(f"Command '{extractor_path}' not found. Trying next command...")
            
        if not extraction_success:
            print("All extractor commands failed. Feature extraction unsuccessful.")
            raise RuntimeError("Feature extraction failed with all available commands")
        
        print(f"Feature extraction successful. Output saved to {output_json_path}")

    except FileNotFoundError as fnf_err:
         # Handle case where extractor or config is not found
         print(f"Error during extraction setup: {fnf_err}")
         print("Please ensure extractor path and config path are correct.")
    except subprocess.CalledProcessError as proc_err:
         # Error message already printed above
         pass # Just proceed to finally block for cleanup
    except Exception as e:
        print(f"An unexpected error occurred during feature extraction: {e}")
    finally:
        # --- 3. Cleanup ---
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                print(f"Step 4: Cleaning up temporary file: {temp_wav_path}")
                os.remove(temp_wav_path)
            except OSError as e:
                print(f"Error cleaning up temporary file {temp_wav_path}: {e}")

    print(f"--- Finished processing: {input_audio} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare audio (mono, 44.1kHz) and extract features using Essentia."
    )
    parser.add_argument("input_audio", help="Path to the input audio file (e.g., MP3, FLAC, WAV).")
    parser.add_argument("output_folder", help="Path to the folder where the output JSON features will be saved.")
    parser.add_argument("--temp_dir", default="extraction_module/tmp", help="Directory for temporary WAV files (default: system temp).")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz).")
    parser.add_argument("--extractor_config", default="pr.yaml",
                        help="Path to the extractor configuration profile (e.g., pr.yaml).")
    args = parser.parse_args()

    # Create directories if they don't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)
        print(f"Created temporary directory: {args.temp_dir}")


    prepare_and_extract(args.input_audio, args.output_folder, args.temp_dir, args.sr, args.extractor_config)