import os
import subprocess

# This script is a "sanity check". It has NO TensorFlow or librosa.
# Its only purpose is to see if Python's subprocess can call ffmpeg on your
# audio files without crashing in a clean environment.

DATASET_PATH = os.path.join('..', 'dataset')
TARGET_SR = 44100
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg')

print("--- Starting FFmpeg Sanity Check ---")
print("This script will try to convert every audio file in your dataset using FFmpeg.")
print("If this script completes, your files and ffmpeg are fine.")
print("If this script crashes, there is a problem with the last file it printed.")

try:
    all_class_labels = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for label in all_class_labels:
        class_dir = os.path.join(DATASET_PATH, label)
        sample_files = [f for f in os.listdir(class_dir) if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)]
        
        print(f"\nProcessing folder: {label} ({len(sample_files)} files)")
        
        for i, filename in enumerate(sample_files):
            file_path = os.path.join(class_dir, filename)
            
            print(f"  [{i+1}/{len(sample_files)}] Checking: {filename}", flush=True)

            command = [
                'ffmpeg',
                '-i', file_path,
                '-f', 's16le',
                '-ac', '1',
                '-ar', str(TARGET_SR),
                '-loglevel', 'panic', # Be very quiet, only show fatal errors
                '-'
            ]
            
            # Run the command. If ffmpeg fails on a file, it will raise an error.
            # A segfault here would be very significant.
            subprocess.run(command, capture_output=True, check=True)

    print("\n\n--- SANITY CHECK PASSED! ---")
    print("All audio files were successfully processed by FFmpeg.")
    print("This confirms the crash is caused by a library conflict with TensorFlow in the main program.")

except subprocess.CalledProcessError as e:
    print("\n\n--- FFMPEG ERROR! ---")
    print(f"FFmpeg failed on the last file printed above.")
    print("This file might be genuinely corrupted in a way that crashes FFmpeg.")

except Exception as e:
    print(f"\n\n--- AN UNEXPECTED ERROR OCCURRED ---")
    print(e)