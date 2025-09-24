import os
import subprocess
import numpy as np
import librosa
import time

# --- THIS SCRIPT HAS NO TENSORFLOW ---

# --- Configuration (Copied from the other script) ---
DATASET_PATH = os.path.join('..', 'dataset')
OUTPUT_FILE = 'preprocessed_data.npz'
AUDIO_TARGET_LENGTH = 2
TARGET_SR = 44100
MIN_SAMPLES_PER_CLASS = 10
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg')

def extract_features_ffmpeg(file_path, n_mfcc=13):
    """Uses FFmpeg to load and standardize audio, then extracts features."""
    try:
        command = [
            'ffmpeg', '-i', file_path, '-f', 's16le', '-ac', '1',
            '-ar', str(TARGET_SR), '-loglevel', 'error', '-'
        ]
        proc = subprocess.run(command, capture_output=True, check=True)
        raw_audio = proc.stdout
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        target_samples = AUDIO_TARGET_LENGTH * TARGET_SR
        audio_np = librosa.util.fix_length(data=audio_np, size=target_samples)
        mfccs = librosa.feature.mfcc(y=audio_np, sr=TARGET_SR, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"      -> SKIPPING: Could not process file: {os.path.basename(file_path)}")
        return None

def main():
    """Main function to run the preprocessing."""
    features, labels = [], []
    
    all_class_labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    valid_class_labels = []
    print("\n--- STAGE 1: PRE-PROCESSING ---")
    print("\n--- Analyzing Dataset ---")
    for label in all_class_labels:
        class_dir = os.path.join(DATASET_PATH, label)
        sample_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)]
        num_samples = len(sample_files)
        if num_samples < MIN_SAMPLES_PER_CLASS:
            print(f"[WARNING] Skipping class '{label}': Only found {num_samples} valid audio samples.")
        else:
            print(f"Class '{label}': Found {num_samples} valid audio samples. OK.")
            valid_class_labels.append((label, sample_files))
    
    print(f"\n--- Starting Data Extraction via FFmpeg ({time.strftime('%H:%M:%S')}) ---")
    
    for label, sample_files in valid_class_labels:
        print(f"\nProcessing folder: {label}")
        for filename in sample_files:
            file_path = os.path.join(DATASET_PATH, label, filename)
            data = extract_features_ffmpeg(file_path)
            if data is not None:
                features.append(data)
                labels.append(label)

    if not features:
        print("ERROR: No features were extracted. Aborting.")
        return

    # Save the processed data to a single, efficient file
    np.savez(OUTPUT_FILE, X=np.array(features), y=np.array(labels))
    print(f"\n--- Pre-processing Complete ({time.strftime('%H:%M:%S')}) ---")
    print(f"Successfully extracted features from {len(features)} files.")
    print(f"Data saved to '{OUTPUT_FILE}'. You can now run the training script.")

if __name__ == '__main__':
    main()