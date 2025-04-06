#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile

def is_audio_file(filename):
    """Checks if a filename has a common audio extension."""
    # Added more extensions
    return os.path.splitext(filename)[1].lower() in {
        '.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.aif'
    }

def batch_prepare_extract(input_folder, output_folder, temp_dir, sr=44100,
                          extractor_config="pr.yaml", prepare_script_path=None):
    """
    Processes all audio files in input_folder using prepare_extract.py.

    Parameters:
      input_folder (str): Path to the folder containing input audio files.
      output_folder (str): Path to the folder where output JSONs will be saved.
      temp_dir (str): Directory for temporary WAV files (passed to prepare_extract.py).
      sr (int): Target sample rate (passed to prepare_extract.py).
      extractor_config (str): Path to extractor config (passed to prepare_extract.py).
      prepare_script_path (str): Path to the prepare_extract.py script itself.
    """
    # If no specific path provided, try multiple locations
    if prepare_script_path is None:
        # First try in the same directory as this script
        script_dir = os.path.dirname(os.path.realpath(__file__))
        possible_paths = [
            os.path.join(script_dir, "prepare_extract.py"),  # Same directory as this script
            "./prepare_extract.py",                          # Current working directory
            os.path.join(script_dir, "../prepare_extract.py"), # Parent directory
            "/workspace/extraction_module/prepare_extract.py"  # Docker container path
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                prepare_script_path = path
                print(f"Found prepare_extract.py at: {path}")
                break
        else:
            # If the loop completes without breaking, no file was found
            print("Error: Could not find prepare_extract.py in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    elif not os.path.isfile(prepare_script_path):
        print(f"Error: The script '{prepare_script_path}' was not found.")
        sys.exit(1)

    # Ensure the output and temp folders exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True) # prepare_extract also does this, but good practice here too

    print(f"Starting batch processing...")
    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Temporary Directory: {temp_dir}")
    print(f"Config File: {extractor_config}")
    print(f"Prepare Script: {prepare_script_path}")
    print("-" * 30)

    processed_count = 0
    error_count = 0

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)

        # Process only if it's a file and looks like an audio file
        if os.path.isfile(input_file_path) and is_audio_file(filename):
            print(f"\nProcessing file: {filename}")

            # Construct the command to run prepare_extract.py
            command = [
                sys.executable, # Use the same python interpreter that's running this script
                prepare_script_path,
                input_file_path,
                output_folder,
                "--temp_dir", temp_dir,
                "--sr", str(sr),
                "--extractor_config", extractor_config
            ]

            print(f"Running command: {' '.join(command)}")

            # Execute the command
            try:
                # Use subprocess.run, capture output, and check return code
                result = subprocess.run(command, capture_output=True, text=True, check=False)

                # Print stdout/stderr from the subscript for better debugging
                if result.stdout:
                    print("--- Subprocess STDOUT ---")
                    print(result.stdout.strip())
                    print("-------------------------")
                if result.stderr:
                    print("--- Subprocess STDERR ---")
                    print(result.stderr.strip())
                    print("-------------------------")


                if result.returncode == 0:
                    print(f"Successfully processed: {filename}")
                    processed_count += 1
                else:
                    print(f"Error processing {filename}. Return code: {result.returncode}")
                    error_count += 1

            except Exception as e:
                print(f"Failed to run subprocess for {filename}: {e}")
                error_count += 1
        elif os.path.isfile(input_file_path):
            print(f"Skipping non-audio file: {filename}")
        # else: it's a directory, ignore

    print("\n" + "=" * 30)
    print("Batch processing finished.")
    print(f"Successfully processed files: {processed_count}")
    print(f"Files with errors: {error_count}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process audio files using prepare_extract.py."
    )
    parser.add_argument("input_folder", help="Path to the folder containing audio files to process.")
    parser.add_argument("output_folder", help="Path to the folder where processed JSON feature files will be saved.")
    parser.add_argument("--prepare_script", default=None,  # Change default to None
                        help="Path to the prepare_extract.py script (default: auto-detect).")
    parser.add_argument("--temp_dir", default=tempfile.gettempdir(),
                        help="Directory for temporary WAV files (default: system temp).")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Target sample rate (default: 44100 Hz).")
    parser.add_argument("--extractor_config", default="extraction_module/pr.yaml",
                        help="Path to the extractor configuration profile (e.g., pr.yaml).")
    args = parser.parse_args()

    batch_prepare_extract(args.input_folder, args.output_folder, args.temp_dir,
                          args.sr, args.extractor_config, args.prepare_script)