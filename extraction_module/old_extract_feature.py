#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import json
import essentia.standard as es
import numpy as np
import traceback

def serialize_value(val):
    """
    Recursively convert Essentia outputs (e.g., numpy arrays, lists, dicts)
    into types that can be serialized by json.
    """
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, list):
        return [serialize_value(item) for item in val]
    elif isinstance(val, dict):
        return dict((k, serialize_value(v)) for k, v in val.items())
    else:
        try:
            return float(val)
        except (TypeError, ValueError):
            return str(val)

def serialize_pool(pool):
    """
    Manually convert an Essentia Pool into a JSON-serializable dictionary.
    """
    # Use the built-in descriptorNames() method from Essentia's Pool object
    pool_dict = pool.descriptorNames()
    
    serialized = {}
    for key in pool_dict:
        serialized[key] = serialize_value(pool[key])
    return serialized

def extract_music_features(input_file, sample_rate=44100):
    """
    Extract music features from an audio file using Essentia's basic MusicExtractor.
    
    Parameters:
      input_file (str): Path to the audio file.
      sample_rate (int): Target sample rate (default 44100 Hz).
      
    Returns:
      features (dict): A dictionary containing extracted music features.
    """
    # Use MusicExtractor directly with the file path
    extractor = es.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                 rhythmStats=['mean', 'stdev'],
                                 tonalStats=['mean', 'stdev'])
    
    # Pass the file path directly to the extractor
    features, features_frames = extractor(input_file)
    
    # Try using the built-in toDict method which is the proper way to convert a Pool to dict
    try:
        return serialize_value(features.toDict())
    except AttributeError:
        # Fall back to our manual method if toDict() is not available
        return serialize_pool(features)

def main():
    parser = argparse.ArgumentParser(
        description="Extract music features using Essentia's basic MusicExtractor (no SVM) and save as JSON."
    )
    parser.add_argument("input", help="Path to the input audio file (e.g., MP3, WAV, FLAC)")
    parser.add_argument("output", help="Path to the output JSON file for extracted features")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate (default: 44100 Hz)")
    args = parser.parse_args()
    
    try:
        features = extract_music_features(args.input, sample_rate=args.sr)
    except Exception as e:
        # print "Error extracting features: {0}".format(e)
        traceback.print_exc()
        return

    with open(args.output, "w") as f:
        json.dump(features, f, indent=2)
    
    # print "Extracted features saved to {0}".format(args.output)

if __name__ == "__main__":
    main()