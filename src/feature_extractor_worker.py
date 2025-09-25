import sys
import os
import subprocess
import numpy as np
import librosa
import json
import base64

# --- THIS SCRIPT IS A STANDALONE WORKER. IT HAS NO TENSORFLOW. ---

# --- Configuration ---
AUDIO_TARGET_LENGTH = 2
TARGET_SR = 44100

def extract_features_ffmpeg(file_path, n_mfcc=13):
    """The robust ffmpeg feature extractor. This is its only job."""
    try:
        command = [
            'ffmpeg', '-i', file_path, '-f', 's16le', '-ac', '1',
            '-ar', str(TARGET_SR), '-loglevel', 'error', '-'
        ]
        proc = subprocess.run(command, capture_output=True, check=True, timeout=120)
        raw_audio = proc.stdout
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        target_samples = AUDIO_TARGET_LENGTH * TARGET_SR
        audio_np = librosa.util.fix_length(data=audio_np, size=target_samples)
        mfccs = librosa.feature.mfcc(y=audio_np, sr=TARGET_SR, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except subprocess.TimeoutExpired:
        # This is a true hang in ffmpeg
        sys.stderr.write(f"TIMEOUT on file: {file_path}\n")
        return None
    except Exception:
        # Any other conversion error
        sys.stderr.write(f"FAILED to process file: {file_path}\n")
        return None

def main():
    """Reads file paths from command line arguments and returns features as JSON."""
    # The first argument is the script name, so we take the rest
    files_to_process = sys.argv[1:]
    
    results = {
        "success": {},
        "failed": []
    }

    for file_path in files_to_process:
        features = extract_features_ffmpeg(file_path)
        if features is not None:
            # Convert numpy array to a list for JSON and store it
            results["success"][file_path] = features.tolist()
        else:
            results["failed"].append(file_path)
            
    # Print the entire results dictionary as a single JSON string to stdout
    print(json.dumps(results))

if __name__ == '__main__':
    main()