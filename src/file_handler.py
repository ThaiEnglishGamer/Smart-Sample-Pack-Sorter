import os
import shutil
import pandas as pd
from datetime import datetime

# --- Configuration ---
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg'}
FORBIDDEN_PATH_FRAGMENTS = {'/program files/', '/windows/', '/system/', '/download/', '$recycle.bin'}
MAX_SCAN_DEPTH = 7
LOG_FILE_NAME = "sample_sorter_log.csv"

def scan_directory(root_directory):
    """
    Scans a directory for audio files with verbose terminal output.
    """
    audio_files, log_file_found = [], False
    root_directory = os.path.normpath(root_directory)
    root_depth = root_directory.count(os.sep)

    print("\n--- STARTING DIRECTORY SCAN ---")
    for current_dir, subdirs, files in os.walk(root_directory, topdown=True):
        if os.path.basename(current_dir) in ["_Unprocessable", "_Sorted Samples"]:
            subdirs[:] = []
            continue
        current_depth = current_dir.count(os.sep)
        if current_depth - root_depth >= MAX_SCAN_DEPTH:
            subdirs[:] = []
            continue
        normalized_current_dir = os.path.normpath(current_dir).lower()
        if any(frag in normalized_current_dir for frag in FORBIDDEN_PATH_FRAGMENTS):
            subdirs[:] = []
            continue
            
        for filename in files:
            full_path = os.path.join(current_dir, filename)
            if current_dir == root_directory and filename == LOG_FILE_NAME:
                log_file_found = True
            
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                print(f"  [SCAN] Found audio file: {filename}")
                audio_files.append(full_path)

    print(f"--- SCAN COMPLETE --- \nFound {len(audio_files)} audio files.")
    return {"audio_files": audio_files, "log_file_found": log_file_found}

def sort_files_transactional(root_directory, predictions, progress_callback, stop_event):
    """
    Performs a fail-safe, transactional sort operation. Logs each action
    before it happens, allowing for recovery from interruptions.
    """
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_folders = set()
    
    # --- Step 1: Create the log file immediately with all pending actions ---
    print("\n--- STARTING SORT OPERATION: Creating initial transaction log ---")
    pending_logs = []
    for original_path, category in predictions:
        pending_logs.append({
            "timestamp": timestamp,
            "original_path": original_path,
            "new_path": os.path.join(root_directory, category, os.path.basename(original_path)),
            "status": "pending"
        })

    log_df = pd.DataFrame(pending_logs)
    write_mode = 'w' if not os.path.exists(log_file_path) else 'a'
    header = not os.path.exists(log_file_path)
    log_df.to_csv(log_file_path, mode=write_mode, header=header, index=False)
    print("Transaction log created. Starting file movement.")

    # --- Step 2: Iterate through the predictions and move files, updating the log live ---
    total_files = len(predictions)
    for i, (original_path, category) in enumerate(predictions):
        if stop_event.is_set():
            print("Stop signal received. Halting file movement.")
            return "Sort aborted by user."

        new_path = os.path.join(root_directory, category, os.path.basename(original_path))
        progress_callback(i + 1, total_files, f"Moving: {os.path.basename(original_path)}")

        try:
            original_folders.add(os.path.dirname(original_path))
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            print(f"  [MOVE] '{os.path.basename(original_path)}' -> '{category}/'")
            shutil.move(original_path, new_path)
            
            # --- Update log for this specific file to 'moved' ---
            df = pd.read_csv(log_file_path)
            df.loc[(df['timestamp'] == timestamp) & (df['original_path'] == original_path), 'status'] = 'moved'
            df.to_csv(log_file_path, index=False)

        except Exception as e:
            print(f"  [ERROR] Could not move file {original_path}. Error: {e}")
            df = pd.read_csv(log_file_path)
            df.loc[(df['timestamp'] == timestamp) & (df['original_path'] == original_path), 'status'] = f'error: {e}'
            df.to_csv(log_file_path, index=False)

    # --- Step 3: Clean up old, empty directories ---
    print("\n--- CLEANUP PHASE ---")
    for folder in sorted(list(original_folders), key=len, reverse=True):
        try:
            if not os.listdir(folder) and folder != root_directory:
                os.rmdir(folder)
                print(f"  [CLEAN] Removed empty folder: {folder}")
        except OSError:
            pass # Ignore folders that can't be removed (e.g., contain hidden files)

    return f"Sort complete. Processed {total_files} files."

def revert_last_sort(root_directory):
    """
    Reads the log file and safely reverts only the files that were
    successfully moved in the last operation.
    """
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    if not os.path.exists(log_file_path):
        return "Error: Log file not found. Cannot revert."
    try:
        print("\n--- STARTING REVERT OPERATION ---")
        df = pd.read_csv(log_file_path)
        last_timestamp = df['timestamp'].iloc[-1]
        
        # --- IMPORTANT: Only revert files with status 'moved' ---
        files_to_revert = df[(df['timestamp'] == last_timestamp) & (df['status'] == 'moved')]

        print(f"Found {len(files_to_revert)} files to revert from session: {last_timestamp}")
        for index, row in files_to_revert.iterrows():
            try:
                original_dir = os.path.dirname(row['original_path'])
                os.makedirs(original_dir, exist_ok=True)
                print(f"  [REVERT] '{os.path.basename(row['new_path'])}' -> '{row['original_path']}'")
                shutil.move(row['new_path'], row['original_path'])
            except FileNotFoundError:
                print(f"  [WARN] Could not find file to revert: {row['new_path']}. It may have been moved or deleted.")
            except Exception as e:
                print(f"  [ERROR] Could not revert {row['new_path']}. Error: {e}")
        
        df_remaining = df[df['timestamp'] != last_timestamp]
        if df_remaining.empty:
            os.remove(log_file_path)
            print("Log file is now empty and has been removed.")
        else:
            df_remaining.to_csv(log_file_path, index=False)
            print("Reverted entries removed from the log.")
        
        return f"Revert complete for session: {last_timestamp}."
    except Exception as e:
        return f"An error occurred during revert: {e}"

def isolate_unprocessable_file(root_directory, file_path):
    """Moves a file that caused a timeout into a dedicated '_Unprocessable' folder."""
    try:
        unprocessable_dir = os.path.join(root_directory, "_Unprocessable")
        os.makedirs(unprocessable_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        destination = os.path.join(unprocessable_dir, filename)
        print(f"\n--- ISOLATING HUNG FILE --- \nMoving {filename} to {unprocessable_dir}\n")
        shutil.move(file_path, destination)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not isolate unprocessable file {file_path}. Error: {e}")