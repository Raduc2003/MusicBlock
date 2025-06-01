import logging
import os
import sys
import subprocess
import tempfile
import shutil # For cleaning up temp directories
from pathlib import Path
import json
from typing import Dict # For loading the output JSON

import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
import uvicorn
import essentia.standard as es # Assuming essentia is installed
import numpy as np

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (can be moved to a config file or env vars later) ---
TARGET_SR = 44100
# This path needs to be valid *inside the Docker container* where this API will run.
EXTRACTOR_CONFIG_FILE = "/app/pr.yaml" # Example path inside the container
# Path to the directory where uploaded files will be temporarily stored *inside the container*
UPLOAD_DIR_IN_CONTAINER = "/app/temp_uploads"
# Path to the directory for Essentia's temporary output *inside the container*
ESSENTIA_OUTPUT_DIR_IN_CONTAINER = "/app/temp_essentia_output"


app = FastAPI(
    title="Feature Extractor API",
    description="Extracts audio features using Essentia.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    # Create temporary directories if they don't exist when the app starts
    os.makedirs(UPLOAD_DIR_IN_CONTAINER, exist_ok=True)
    os.makedirs(ESSENTIA_OUTPUT_DIR_IN_CONTAINER, exist_ok=True)
    logger.info(f"Temporary upload directory: {os.path.abspath(UPLOAD_DIR_IN_CONTAINER)}")
    logger.info(f"Temporary Essentia output directory: {os.path.abspath(ESSENTIA_OUTPUT_DIR_IN_CONTAINER)}")
    # Check if extractor config exists
    if not os.path.isfile(EXTRACTOR_CONFIG_FILE):
        logger.error(f"CRITICAL: Extractor config file '{EXTRACTOR_CONFIG_FILE}' not found. API may not function correctly.")


def _cleanup_temp_files(*paths):
    for path in paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"Cleaned up temp file: {path}")
            elif os.path.isdir(path): 
                shutil.rmtree(path)
                logger.info(f"Cleaned up temp directory: {path}")
        except Exception as e:
            logger.error(f"Error cleaning up {path}: {e}")


def run_feature_extraction_logic(  # Remove 'async' here
    input_audio_path: str, # Path to the saved uploaded file
    base_filename_no_ext: str # e.g., "my_song"
) -> Dict: # Returns the content of the JSON feature file
    """
    Encapsulates the core feature extraction logic from prepare_extract.py.
    Manages its own temporary files for intermediate WAV and output JSON.
    """
    temp_wav_path = None
    output_json_path = None 

    try:
        logger.info(f"Feature Extractor: Starting processing for {input_audio_path}")

        # --- 1. Prepare Audio (Load, Resample, Mono) ---
        logger.info(f"Step 1: Loading, resampling to {TARGET_SR} Hz, converting to mono...")
        loader = es.MonoLoader(filename=input_audio_path, sampleRate=TARGET_SR)
        audio = loader()

        if np.max(np.abs(audio)) == 0.0:
            logger.warning(f"Input file {input_audio_path} appears silent. Skipping actual Essentia extraction.")
            # Return a default or empty feature structure if needed by downstream
            return {"error": "Audio file is silent", "features": None}

        # Create a temporary WAV file for Essentia to process
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="prep_", dir=UPLOAD_DIR_IN_CONTAINER, delete=False) as tmp_wav_file:
            temp_wav_path = tmp_wav_file.name
        
        logger.info(f"Step 2: Saving temporary processed WAV to {temp_wav_path}...")
        writer = es.MonoWriter(filename=temp_wav_path, sampleRate=TARGET_SR, format='wav')
        writer(audio)
        logger.info("Temporary WAV saved successfully.")

        # --- 2. Extract Features ---
        # Essentia will write its output JSON to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".json", prefix=f"features_{base_filename_no_ext}_", dir=ESSENTIA_OUTPUT_DIR_IN_CONTAINER, delete=False) as tmp_json_file:
            output_json_path = tmp_json_file.name

        logger.info("Step 3: Running Essentia extractor...")
        logger.info(f"  Input (temp WAV): {temp_wav_path}")
        logger.info(f"  Output (temp JSON): {output_json_path}")
        logger.info(f"  Config: {EXTRACTOR_CONFIG_FILE}")

        if not os.path.isfile(EXTRACTOR_CONFIG_FILE):
            raise FileNotFoundError(f"Extractor config not found at container path: {EXTRACTOR_CONFIG_FILE}")

        extractor_commands = ["essentia_streaming_extractor_music", "streaming_extractor_music"]
        extraction_success = False
        
        for extractor_exe in extractor_commands:
            logger.info(f"Trying extractor: {extractor_exe}")
            cmd = [extractor_exe, temp_wav_path, output_json_path, EXTRACTOR_CONFIG_FILE]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60) # Added timeout
                if result.returncode == 0:
                    logger.info(f"Feature extraction successful using '{extractor_exe}'.")
                    extraction_success = True
                    break
                else:
                    logger.error(f"Command '{extractor_exe}' failed. RC: {result.returncode}\nStderr: {result.stderr}\nStdout: {result.stdout}")
            except FileNotFoundError:
                logger.warning(f"Command '{extractor_exe}' not found.")
            except subprocess.TimeoutExpired:
                logger.error(f"Command '{extractor_exe}' timed out after 60 seconds.")
        
        if not extraction_success:
            raise RuntimeError("Feature extraction failed with all available commands.")
        
        logger.info(f"Feature extraction output saved to temp JSON: {output_json_path}")

        # Load the features from the JSON file
        with open(output_json_path, 'r') as f:
            features_data = json.load(f)
        
        return features_data # Return the content of the JSON

    except Exception as e:
        logger.error(f"Error during feature extraction for {input_audio_path}: {e}", exc_info=True)
        # Ensure we pass a dict that can be JSON serialized if an error occurs before features_data is set
        return {"error": str(e), "features": None} 
    finally:
        # Cleanup all temporary files created by this request
        _cleanup_temp_files(temp_wav_path, output_json_path)


@app.post("/extract_features")
async def extract_features_endpoint(background_tasks: BackgroundTasks, audio_file: UploadFile = File(...)):
    temp_uploaded_audio_path = None # Initialize to ensure it's always defined for cleanup
    try:
        # Use a unique name for the uploaded file within the UPLOAD_DIR_IN_CONTAINER
        # Create the temp file first, then write to it
        with tempfile.NamedTemporaryFile(suffix=Path(audio_file.filename).suffix if audio_file.filename else ".tmp", dir=UPLOAD_DIR_IN_CONTAINER, delete=False) as tmp_file:
            temp_uploaded_audio_path = tmp_file.name
        
        logger.info(f"Receiving file: {audio_file.filename}. Saving to temp path: {temp_uploaded_audio_path}")
        with open(temp_uploaded_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"File {audio_file.filename} saved to {temp_uploaded_audio_path}")
        
        # Ensure the uploaded file is cleaned up in the background
        background_tasks.add_task(_cleanup_temp_files, temp_uploaded_audio_path)

        base_filename = Path(audio_file.filename).stem if audio_file.filename else "audio_features"

        logger.info(f"Calling run_feature_extraction_logic in thread pool for {base_filename}...")
        
        # feature_json_content should be the dictionary returned by run_feature_extraction_logic
        feature_json_content = await fastapi.concurrency.run_in_threadpool(
            run_feature_extraction_logic, # This is your synchronous function
            input_audio_path=temp_uploaded_audio_path, 
            base_filename_no_ext=base_filename
        )
        
        logger.info(f"Received result from thread pool. Type: {type(feature_json_content)}")
        logger.debug(f"Full feature_json_content from thread pool: {feature_json_content}")


        # Now check the actual content
        if isinstance(feature_json_content, dict): # Explicitly check if it's a dict
            if "error" in feature_json_content and feature_json_content.get("error") == "Audio file is silent":
                logger.warning(f"Processed silent audio file: {audio_file.filename}")
                raise HTTPException(status_code=400, detail=f"Audio file '{audio_file.filename}' is silent, no features extracted.")
            elif "error" in feature_json_content and feature_json_content.get("error") is not None: # Any other error string
                logger.error(f"Feature extraction failed for {audio_file.filename}. Error in returned dict: {feature_json_content['error']}")
                raise HTTPException(status_code=500, detail=f"Feature extraction failed: {feature_json_content['error']}")
            elif "error" not in feature_json_content and feature_json_content: # Success case
                 logger.info(f"Successfully extracted features for {audio_file.filename}.")
                 return feature_json_content
            else: # Empty dict or other unexpected dict content
                logger.error(f"Feature extraction returned unexpected dictionary for {audio_file.filename}: {feature_json_content}")
                raise HTTPException(status_code=500, detail="Feature extraction returned unexpected data.")
        else:
            # This case should ideally not be hit if run_in_threadpool works as expected
            logger.error(f"Feature extraction did not return a dictionary for {audio_file.filename}. Got type: {type(feature_json_content)}")
            raise HTTPException(status_code=500, detail="Feature extraction failed to produce valid output.")

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # No need to add temp_uploaded_audio_path to background_tasks again here,
        # it was added before the try-block that might fail (run_in_threadpool)
        logger.error(f"Unhandled error in /extract_features endpoint for {audio_file.filename if audio_file else 'unknown file'}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during feature extraction: {str(e)}")


if __name__ == "__main__":
    # This allows running the API directly with Uvicorn for local testing.
    # Example: python similarity_pipeline/feature_extractor_api.py
    # The Dockerfile CMD will also use uvicorn.
    
    # Ensure the extractor config is discoverable if running locally for testing
    # This might mean placing pr.yaml relative to this script or adjusting EXTRACTOR_CONFIG_FILE path.
    # For local test, assume pr.yaml is in the same directory or a known relative path.
    # If EXTRACTOR_CONFIG_FILE is "/app/pr.yaml", you need to ensure it's there when testing locally without Docker.
    # For instance, create a symlink or copy it.
    
    # Make sure EXTRACTOR_CONFIG_FILE is set correctly or exists where expected
    # For local test, you might do:
    # if not os.path.exists(EXTRACTOR_CONFIG_FILE) and os.path.exists("pr.yaml"):
    #     EXTRACTOR_CONFIG_FILE = "pr.yaml" # Use local pr.yaml if available
    
    logger.info(f"Starting Feature Extractor API server with Uvicorn on 0.0.0.0:8001 (Example Port)...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") # Using port 8001 for this service