#!/usr/bin/env python
import argparse
import json
import numpy as np
import essentia.standard as es

def extract_features(audio_file):
    # Load audio: convert to mono and resample to 44100 Hz
    loader = es.MonoLoader(filename=audio_file, sampleRate=44100)
    audio = loader()

    # -----------------------------------
    # 1. Compute MFCCs: extract frame-wise MFCCs and compute the mean of the first 13 coefficients
    # -----------------------------------
    frameCutter = es.FrameCutter(frameSize=2048, hopSize=1024)
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc_extractor = es.MFCC(numberCoefficients=13)
    
    mfcc_list = []
    for frame in frameCutter(audio):
        # Ensure frame is array-like and long enough
        if not hasattr(frame, "__len__") or len(frame) < 2048:
            continue
        frame_windowed = window(frame)
        spec = spectrum(frame_windowed)
        _, mfcc_coeffs = mfcc_extractor(spec)
        mfcc_list.append(mfcc_coeffs)
    mfcc_mean = np.mean(mfcc_list, axis=0).tolist()

    # -----------------------------------
    # 2. Extract Rhythm: BPM and beat positions using RhythmExtractor2013 (Degara method)
    # -----------------------------------
    rhythmExtractor = es.RhythmExtractor2013(method="degara")
    rhythm_result = rhythmExtractor(audio)
    # Use only the first four values (bpm, beats, beats_confidence, beats_intervals)
    bpm = rhythm_result[0]
    beats = rhythm_result[1]
    beats_confidence = rhythm_result[2]
    beats_intervals = rhythm_result[3]

    # -----------------------------------
    # 3. Compute HPCP: extract frame-wise HPCP and compute the mean vector
    # -----------------------------------
    frameCutter2 = es.FrameCutter(frameSize=4096, hopSize=2048)
    window2 = es.Windowing(type="blackmanharris62")
    spectrum2 = es.Spectrum()
    hpcp_extractor = es.HPCP()

    hpcp_list = []
    for frame in frameCutter2(audio):
        if not hasattr(frame, "__len__") or len(frame) < 4096:
            continue
        frame_windowed2 = window2(frame)
        spec2 = spectrum2(frame_windowed2)
        hpcp_vector = hpcp_extractor(spec2)
        hpcp_list.append(hpcp_vector)
    hpcp_mean = np.mean(hpcp_list, axis=0).tolist()

    # -----------------------------------
    # 4. Extract Key: compute key, scale, and key strength
    # -----------------------------------
    key_extractor = es.Key()
    key, scale, key_strength = key_extractor(audio)

    # -----------------------------------
    # Combine all features into a dictionary
    # -----------------------------------
    features = {
        "mfcc_mean": mfcc_mean,       # 13-dimensional vector
        "bpm": bpm,                   # Beats per minute
        "beats": beats,               # Beat positions (seconds)
        "hpcp_mean": hpcp_mean,       # Mean HPCP vector
        "key": key,                   # Detected key (e.g., "C")
        "scale": scale,               # Detected scale ("major" or "minor")
        "key_strength": key_strength  # Key strength score
    }
    return features

def main():
    parser = argparse.ArgumentParser(
        description="Extract audio features using Essentia: MFCC mean, rhythm (BPM & beats), HPCP mean, and key information."
    )
    parser.add_argument("audio_file", help="Path to the input audio file (e.g., WAV, MP3, FLAC)")
    parser.add_argument("output_file", help="Path to save the extracted features as JSON")
    args = parser.parse_args()

    features = extract_features(args.audio_file)
    with open(args.output_file, "w") as f:
        json.dump(features, f, indent=4)
    
    print(f"Features extracted and saved to {args.output_file}")

if __name__ == "__main__":
    main()
