import subprocess
import os
import argparse
import sys

def is_audio_file(filename):
    # Check for common audio file extensions
    return os.path.splitext(filename)[1].lower() in {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac'}


def main():
    parser = argparse.ArgumentParser(
        description="Batch process audio files using prepare_audio.py for loudness normalization."
    )
    parser.add_argument("input_folder", help="Path to the folder containing audio files to process")
    parser.add_argument("output_folder", help="Path to the folder where processed files will be saved")
    args = parser.parse_args()

    # Ensure the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # List all files in the input folder and filter by audio extensions
    for filename in os.listdir(args.input_folder):
        if is_audio_file(filename):
            input_file = os.path.join(args.input_folder, filename)
            # Save processed files as WAV; adjust filename if needed
            output_filename = "output_" + os.path.splitext(filename)[0] + ".json"
            output_file = os.path.join(args.output_folder, output_filename)
            print("Processing: {} -> {}".format(input_file, output_file))

            # Try different possible executable names
            possible_executables = [
                "essentia_streaming_extractor_music",
                "essentia_streaming_extractor_music.py",
                "/usr/local/bin/essentia_streaming_extractor_music",
                "/usr/bin/essentia_streaming_extractor_music"
            ]
            
            success = False
            for executable in possible_executables:
                print("Trying executable: {}".format(executable))
                try:
                    return_code = subprocess.call([
                        executable,
                        input_file,
                        output_file,
                        "pr.yaml"
                    ])
                    if return_code == 0:
                        success = True
                        print("Successfully processed with: {}".format(executable))
                        break
                    else:
                        print("Error processing {}: return code {}".format(input_file, return_code))
                except OSError as e:
                    if e.errno == 2:  # No such file or directory
                        print("Executable not found: {}".format(executable))
                    else:
                        print("Error executing process: {}".format(e))
            
            if not success:
                print("Failed to process {} with any available executable.".format(input_file))
                print("Debug info: Python version: {}, Path: {}".format(
                    sys.version, os.environ.get('PATH', 'PATH not available')
                ))
                print("Try running 'which essentia_streaming_extractor_music' in your shell to locate the executable")

if __name__ == "__main__":
    main()
