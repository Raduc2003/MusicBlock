import argparse
import librosa
import soundfile as sf
import pyloudnorm as pyln
import numpy as np

def prepare_audio(input_path, output_path, target_lufs=-23.0, sr=44100):
    """
    Prepares an audio file for extraction by:
      - Loading and converting to mono at the target sample rate
      - Measuring integrated loudness using EBU R128
      - Applying gain to adjust loudness to the target LUFS
      - Writing out the processed audio in WAV format
    
    Parameters:
      input_path (str): Path to the input audio file.
      output_path (str): Path where the processed file will be saved.
      target_lufs (float): Desired integrated loudness (default -23 LUFS)
      sr (int): Target sample rate (default 44100 Hz)
    """
    # Load audio (ensures mono and resamples to 44100 Hz)
    audio, _ = librosa.load(input_path, sr=sr, mono=True)
    
    # Create a loudness meter (pyloudnorm expects a numpy array and sample rate)
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)
    # print(f"Current integrated loudness: {current_loudness:.2f} LUFS")
    
    # Calculate gain required (in dB) to reach target loudness
    gain_db = target_lufs - current_loudness
    gain = 10 ** (gain_db / 20.0)
    # print(f"Applying gain of {gain_db:.2f} dB (linear factor: {gain:.2f})")
    
    # Apply gain to the audio signal
    processed_audio = audio * gain
    
    # Verify new loudness (optional)
    new_loudness = meter.integrated_loudness(processed_audio)
    # print(f"New integrated loudness: {new_loudness:.2f} LUFS")
    
    # Save the processed audio as WAV
    sf.write(output_path, processed_audio, sr)
    # print(f"Processed audio saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare audio for AcousticBrainz extraction: mono, 44100 Hz, and loudness normalized."
    )
    parser.add_argument("input", help="Path to the input audio file (e.g., MP3, FLAC)")
    parser.add_argument("output", help="Path to save the processed audio file (WAV format recommended)")
    parser.add_argument("--target_lufs", type=float, default=-23.0, help="Target integrated loudness in LUFS (default: -23)")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz)")
    args = parser.parse_args()

    prepare_audio(args.input, args.output, target_lufs=args.target_lufs, sr=args.sr)
