#!/usr/bin/env python3
import os
import argparse
import subprocess

def is_audio_file(filename):
    # Check for common audio file extensions
    return os.path.splitext(filename)[1].lower() in {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac'}

def main():
    parser = argparse.ArgumentParser(
        description="Batch process audio files using prepare_audio.py for loudness normalization."
    )
    parser.add_argument("input_folder", help="Path to the folder containing audio files to process")
    parser.add_argument("output_folder", help="Path to the folder where processed files will be saved")
    parser.add_argument("--target_lufs", type=float, default=-23.0, help="Target integrated loudness (default: -23 LUFS)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz)")
    args = parser.parse_args()

    # Ensure the output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # List all files in the input folder and filter by audio extensions
    for filename in os.listdir(args.input_folder):
        if is_audio_file(filename):
            input_file = os.path.join(args.input_folder, filename)
            # Save processed files as WAV; adjust filename if needed
            output_filename = os.path.splitext(filename)[0] + ".mp3"
            output_file = os.path.join(args.output_folder, output_filename)
            print(f"Processing: {input_file} -> {output_file}")

            # Call the prepare_audio.py script for each file
            subprocess.run([
                "python", "prepare_audio.py",
                input_file,
                output_file,
                "--target_lufs", str(args.target_lufs),
                "--sr", str(args.sr)
            ])

if __name__ == "__main__":
    main()

