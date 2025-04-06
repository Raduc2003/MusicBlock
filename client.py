#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Chain audio preparation, feature extraction, and similarity search using Essentia."
    )
    parser.add_argument("input_audio", help="Path to the input audio file (e.g. MP3, FLAC)")
    parser.add_argument("--temp_dir", default="temp", help="Temporary directory for intermediate files")
    parser.add_argument("--target_lufs", type=float, default=-23.0, help="Target LUFS for audio processing (default: -23)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz)")
    parser.add_argument("--extractor_config", default="pr.yaml", help="Essentia extractor config file (default: pr.yaml)")
    parser.add_argument("--top_k", type=int, default=20, help="Number of similar items to retrieve (passed to client_similarity)")
    args = parser.parse_args()

    try:
        os.makedirs(args.temp_dir)
    except:
        pass

    processed_audio = os.path.join(args.temp_dir, "processed_audio.wav")
    print("=== Preparing Audio ===")
    prepare_cmd = [
        "python", "prepare_audio.py",
        args.input_audio,
        processed_audio,
        "--target_lufs", str(args.target_lufs),
        "--sr", str(args.sr)
    ]
    print("callning command:", " ".join(prepare_cmd))
    result = subprocess.call(prepare_cmd)
    if result != 0:
        print("Error: Audio preparation failed.")
        sys.exit(1)

    output_json = os.path.join(args.temp_dir, "output_features.json")
    print("\n=== Extracting Features ===")
    extractor_cmd = [
        "essentia_streaming_extractor_music",
        processed_audio,
        output_json,
        args.extractor_config
    ]
    print("callning command:", " ".join(extractor_cmd))
    result = subprocess.call(extractor_cmd)
    if result != 0:
        print("Error: Feature extraction failed.")
        sys.exit(1)

    print("\n=== Performing Similarity Search ===")
    similarity_cmd = [
        "python", "client_similarity.py",
        output_json,
        "--top_k", str(args.top_k)
    ]
    print("callning command:", " ".join(similarity_cmd))
    result = subprocess.call(similarity_cmd)
    if result != 0:
        print("Error: Similarity search failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
