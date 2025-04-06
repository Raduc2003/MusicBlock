#!/usr/bin/env python3
import argparse
import numpy as np
import essentia.standard as es
import warnings

def prepare_audio(input_path, output_path, target_lufs=-23.0, sr=44100):
    """
    Prepares an audio file using Essentia's LoudnessEBUR128,
    correctly extracting the integrated loudness value from index 2.
    Loads mono, converts to stereo for analysis, applies gain, handles clipping,
    and saves as WAV.
    """
    try:
        # Load audio as mono
        loader = es.MonoLoader(filename=input_path, sampleRate=sr)
        audio = loader()
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return # Cannot proceed

    # Handle silent files
    if np.max(np.abs(audio)) == 0.0:
        warnings.warn(f"Input file {input_path} appears silent. Skipping normalization.")
        try:
            writer = es.MonoWriter(filename=output_path, sampleRate=sr, format='wav')
            writer(audio)
            print(f"Silent audio saved (no normalization) to {output_path}")
        except Exception as e:
             print(f"Error writing silent file {output_path}: {e}")
        return # Exit function early

    # Convert mono to stereo (duplicate channel) for LoudnessEBUR128 input
    stereo_audio = np.column_stack((audio, audio))

    # Use LoudnessEBUR128 - expects stereo input
    loudness_analyzer = es.LoudnessEBUR128() # Use default parameters

    # --- Measure Loudness ---
    try:
        loudness_result = loudness_analyzer(stereo_audio)
        # --- Extract the INTEGRATED loudness (Element at Index 2) ---
        raw_lufs_val = loudness_result[2]
        print(f"DEBUG: Extracted integrated LUFS (loudness_result[2]): {raw_lufs_val}, Type: {type(raw_lufs_val)}")
    except IndexError:
         print(f"Error: LoudnessEBUR128 output tuple doesn't have expected index 2 for {input_path}.")
         raw_lufs_val = None # Mark as invalid
    except Exception as e:
        print(f"Error measuring loudness with LoudnessEBUR128 for {input_path}: {e}")
        raw_lufs_val = None # Mark as invalid

    # Validate the extracted loudness value
    if raw_lufs_val is None or not (np.isscalar(raw_lufs_val) and np.isfinite(raw_lufs_val)):
        warnings.warn(f"Could not determine valid integrated loudness for {input_path} (raw value: {raw_lufs_val}). Skipping normalization.")
        # Save original if loudness calculation failed
        try:
            writer = es.MonoWriter(filename=output_path, sampleRate=sr, format='wav')
            writer(audio)
            print(f"Audio saved without normalization due to invalid LUFS to {output_path}")
        except Exception as e:
             print(f"Error writing non-normalized file {output_path}: {e}")
        return # Exit function early

    current_lufs = float(raw_lufs_val)
    print(f"Current integrated loudness: {current_lufs:.2f} LUFS")

    # --- Calculate and Apply Gain ---
    gain_db = target_lufs - current_lufs
    print(f"Calculated gain adjustment: {gain_db:.2f} dB")

    # Convert gain from dB to a linear factor
    gain_linear = 10 ** (gain_db / 20.0)

    # Apply the gain using float64 for precision
    processed_audio = (audio.astype(np.float64) * gain_linear)

    # --- Clipping Check and Handling ---
    peak_level = np.max(np.abs(processed_audio))
    print(f"DEBUG: Max absolute value after gain: {peak_level:.4f}")
    if peak_level > 1.0:
        warnings.warn(f"Potential clipping detected! Peak level = {peak_level:.4f}. Clamping to [-1.0, 1.0].")
        processed_audio = np.clip(processed_audio, -1.0, 1.0)

    # Convert back to float32 for saving
    processed_audio = processed_audio.astype(np.float32)

    # --- Verification Step (Optional but recommended) ---
    try:
        # For verification, we still need to use LoudnessEBUR128 on stereo
        verify_stereo = np.column_stack((processed_audio, processed_audio))
        verify_loudness_analyzer = es.LoudnessEBUR128()
        verify_result = verify_loudness_analyzer(verify_stereo)

        # Extract integrated LUFS from index 2 again for verification
        final_lufs = float('nan') # Default to NaN
        if len(verify_result) > 2 and np.isscalar(verify_result[2]) and np.isfinite(verify_result[2]):
             final_lufs = float(verify_result[2])

        print(f"Verified integrated loudness: {final_lufs:.2f} LUFS")
        # Check if the result is close to the target
        if not np.isnan(final_lufs) and not (target_lufs - 1.0 <= final_lufs <= target_lufs + 1.0):
             warnings.warn(f"Verification shows final LUFS ({final_lufs:.2f}) is not within +/- 1.0 dB of target ({target_lufs:.1f}).")
    except Exception as e:
        print(f"Warning: Could not verify loudness after processing. Error: {e}")

    # --- Save the Processed Audio ---
    try:
        # Save as WAV using MonoWriter (no bitDepth parameter)
        writer = es.MonoWriter(filename=output_path, sampleRate=sr, format='wav')
        writer(processed_audio)
        print(f"Processed audio saved to {output_path}")
    except Exception as e:
        print(f"Error writing processed file {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare audio: mono, 44100 Hz, loudness normalized using Essentia's LoudnessEBUR128."
    )
    parser.add_argument("input", help="Path to the input audio file (e.g., MP3, FLAC)")
    parser.add_argument("output", help="Path to save the processed audio file (WAV format recommended)")
    parser.add_argument("--target_lufs", type=float, default=-23.0, help="Target integrated loudness in LUFS (default: -23)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz)")
    args = parser.parse_args()

    prepare_audio(args.input, args.output, target_lufs=args.target_lufs, sr=args.sr)