import os
import shutil
import pandas as pd
from datetime import datetime

# --- Configuration ---
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg'}
FORBIDDEN_PATH_FRAGMENTS = {'/program files/', '/windows/', '/system/', '/download/', '$recycle.bin'}
MAX_SCAN_DEPTH = 7
LOG_FILE_NAME = "sample_sorter_log.csv"
# --- REMOVED: We no longer need the SORTED_FOLDER_NAME ---

def scan_directory(root_directory):
    """Scans a directory for audio files."""
    # This function is already perfect and needs no changes.
    audio_files, log_file_found = [], False
    root_directory = os.path.normpath(root_directory)
    root_depth = root_directory.count(os.sep)

    print(f"Starting scan in: {root_directory}")
    for current_dir, subdirs, files in os.walk(root_directory, topdown=True):
        current_depth = current_dir.count(os.sep)
        if current_dir.endswith("_Uncategorized"): # Don't re-scan our own output folders
            subdirs[:] = []
            continue
        if current_depth - root_depth >= MAX_SCAN_DEPTH:
            subdirs[:] = []
            continue
        normalized_current_dir = os.path.normpath(current_dir).lower()
        if any(frag in normalized_current_dir for frag in FORBIDDEN_PATH_FRAGMENTS):
            subdirs[:] = []
            continue
        for filename in files:
            if current_dir == root_directory and filename == LOG_FILE_NAME:
                log_file_found = True
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                full_path = os.path.join(current_dir, filename)
                audio_files.append(full_path)
    print(f"Scan complete. Found {len(audio_files)} audio files.")
    return {"audio_files": audio_files, "log_file_found": log_file_found}


def sort_files(root_directory, predictions):
    """
    Moves files to new categorized folders within the root directory,
    creates a log, and cleans up old empty folders.
    """
    log_data = []
    original_folders = set() # Keep track of where files came from
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for original_path, category in predictions:
        try:
            original_folders.add(os.path.dirname(original_path))
            
            # --- MODIFIED: The destination is now the root directory itself ---
            category_dir = os.path.join(root_directory, category)
            os.makedirs(category_dir, exist_ok=True)
            
            filename = os.path.basename(original_path)
            new_path = os.path.join(category_dir, filename)
            
            shutil.move(original_path, new_path)
            
            log_data.append({
                "timestamp": timestamp, "original_path": original_path,
                "new_path": new_path, "status": "moved"
            })
        except Exception as e:
            print(f"Could not move file {original_path}. Error: {e}")
            log_data.append({
                "timestamp": timestamp, "original_path": original_path,
                "new_path": "N/A", "status": f"error: {e}"
            })

    # --- NEW: Clean up the old, now-empty directories ---
    print("Cleaning up old empty directories...")
    for folder in sorted(list(original_folders), key=len, reverse=True):
        try:
            # Check if folder is empty and not the root itself
            if not os.listdir(folder) and folder != root_directory:
                os.rmdir(folder)
                print(f"Removed empty folder: {folder}")
        except OSError as e:
            print(f"Could not remove folder {folder}. It might not be empty. Error: {e}")


    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    log_df = pd.DataFrame(log_data)
    
    if os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file_path, mode='w', header=True, index=False)

    return f"Sort complete. Processed {len(predictions)} files."

def revert_last_sort(root_directory):
    """
    Reads the log file and moves files back to their original locations.
    (This function is already perfect and needs no changes).
    """
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    if not os.path.exists(log_file_path):
        return "Error: Log file not found. Cannot revert."
    try:
        df = pd.read_csv(log_file_path)
        last_timestamp = df['timestamp'].iloc[-1]
        last_sort_df = df[df['timestamp'] == last_timestamp]
        files_to_revert = last_sort_df[last_sort_df['status'] == 'moved']
        for index, row in files_to_revert.iterrows():
            try:
                original_dir = os.path.dirname(row['original_path'])
                os.makedirs(original_dir, exist_ok=True)
                shutil.move(row['new_path'], row['original_path'])
            except FileNotFoundError:
                print(f"Warning: Could not find file to revert: {row['new_path']}.")
            except Exception as e:
                print(f"Could not revert {row['new_path']}. Error: {e}")
        df_remaining = df[df['timestamp'] != last_timestamp]
        if df_remaining.empty:
            os.remove(log_file_path)
        else:
            df_remaining.to_csv(log_file_path, index=False)
        return f"Revert complete for sort operation from {last_timestamp}."
    except Exception as e:
        return f"An error occurred during revert: {e}"