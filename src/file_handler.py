import os
import shutil
import pandas as pd
from datetime import datetime
import hashlib

# --- Configuration ---
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg'}
LOG_FILE_NAME = "sample_sorter_log.csv"

# --- NEW: Comprehensive map for non-audio files ---
NON_AUDIO_MAP = {
    # DAW Project Files
    '.flp': 'Project Files', '.als': 'Project Files', '.logicx': 'Project Files',
    '.cpr': 'Project Files', '.rpp': 'Project Files', '.song': 'Project Files',
    # VST Presets
    '.fxp': 'VST Presets', '.fxb': 'VST Presets', # General VST2
    '.vstpreset': 'VST Presets', # General VST3
    '.wav': 'Wavetables', # Serum/Vital can use wavs as tables
    '.vital': 'Vital Presets',
    '.fst': 'FL Studio Presets',
    '.nmsv': 'Massive Presets', '.mxprj': 'Massive X Presets',
    '.sbf': 'Spire Presets',
    '.h2p': 'Diva Presets',
    '.pgtx': 'Pigments Presets',
    '.nki': 'Kontakt Instruments', '.nkm': 'Kontakt Instruments', '.nkb': 'Kontakt Instruments',
    # Other
    '.mid': 'MIDI Files',
    '.sf2': 'SoundFonts',
}
OTHER_STUFF_FOLDER = "Other Stuff"

def get_file_hash(path):
    """Calculates the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def scan_directory(root_directory):
    """Scans for ALL files and separates them into audio and non-audio."""
    audio_files, non_audio_files, log_file_found = [], [], False
    root_directory = os.path.normpath(root_directory)
    print("\n--- STARTING FULL DIRECTORY SCAN ---")
    for current_dir, subdirs, files in os.walk(root_directory, topdown=True):
        if os.path.basename(current_dir).startswith('_') or os.path.basename(current_dir) in NON_AUDIO_MAP.values():
            subdirs[:] = []
            continue
        for filename in files:
            full_path = os.path.join(current_dir, filename)
            if current_dir == root_directory and filename == LOG_FILE_NAME:
                log_file_found = True
                continue
            
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.append(full_path)
            else:
                non_audio_files.append(full_path)

    print(f"--- SCAN COMPLETE --- \nFound {len(audio_files)} audio files and {len(non_audio_files)} other files.")
    return {"audio_files": audio_files, "non_audio_files": non_audio_files, "log_file_found": log_file_found}

def organize_library(root_directory, audio_predictions, non_audio_files, progress_callback, stop_event):
    """
    The master function for organizing the entire library, including all file types,
    duplicates, and cleanup, using a fail-safe transactional log.
    """
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_folders = set()
    
    # --- Step 1: Create a comprehensive "to-do list" log file first ---
    print("\n--- STARTING ORGANIZATION: Creating initial transaction log ---")
    pending_logs = []
    all_predictions = audio_predictions.copy()
    
    # Add non-audio files to the to-do list
    for path in non_audio_files:
        ext = os.path.splitext(path)[1].lower()
        category = NON_AUDIO_MAP.get(ext, OTHER_STUFF_FOLDER)
        all_predictions.append((path, category))

    for original_path, category in all_predictions:
        original_folders.add(os.path.dirname(original_path))
        pending_logs.append({
            "timestamp": timestamp, "action": "move",
            "original_path": original_path,
            "new_path": os.path.join(root_directory, category, os.path.basename(original_path)),
            "status": "pending"
        })

    log_df = pd.DataFrame(pending_logs)
    write_mode = 'a' if os.path.exists(log_file_path) else 'w'
    header = not os.path.exists(log_file_path)
    log_df.to_csv(log_file_path, mode=write_mode, header=header, index=False)
    print("Transaction log created. Starting file movement.")

    # --- Step 2: Move all files according to the log ---
    total_files = len(all_predictions)
    for i, (original_path, category) in enumerate(all_predictions):
        if stop_event.is_set(): return "Organization aborted by user."
        
        new_path = os.path.join(root_directory, category, os.path.basename(original_path))
        progress_callback(i, total_files, f"Moving: {os.path.basename(original_path)}")
        
        try:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(original_path, new_path)
            df = pd.read_csv(log_file_path)
            df.loc[(df['timestamp'] == timestamp) & (df['original_path'] == original_path), 'status'] = 'moved'
            df.to_csv(log_file_path, index=False)
        except Exception as e:
            df = pd.read_csv(log_file_path)
            df.loc[(df['timestamp'] == timestamp) & (df['original_path'] == original_path), 'status'] = f'error: {e}'
            df.to_csv(log_file_path, index=False)

    # --- Step 3: Handle Duplicates in the new sorted folders ---
    print("\n--- CHECKING FOR DUPLICATES ---")
    progress_callback(total_files -1, total_files, "Checking for duplicates...")
    potential_dupes = {}
    for current_dir, _, files in os.walk(root_directory):
        if os.path.basename(current_dir).startswith('_') or os.path.basename(current_dir) in NON_AUDIO_MAP.values():
            for filename in files:
                full_path = os.path.join(current_dir, filename)
                potential_dupes.setdefault(filename, []).append(full_path)
    
    hashes = {}
    for filename, paths in potential_dupes.items():
        if len(paths) > 1:
            for path in paths:
                if path not in hashes: hashes[path] = get_file_hash(path)
            
            unique_hashes = {}
            for path in paths:
                file_hash = hashes[path]
                if file_hash not in unique_hashes:
                    unique_hashes[file_hash] = []
                unique_hashes[file_hash].append(path)

            for file_hash, hash_paths in unique_hashes.items():
                if len(hash_paths) > 1: # Perfect duplicates
                    for path_to_delete in hash_paths[1:]:
                        print(f"  [DELETE] Duplicate found: Deleting {os.path.basename(path_to_delete)}")
                        os.remove(path_to_delete)
                        log_entry = pd.DataFrame([{"timestamp": timestamp, "action": "delete", "original_path": path_to_delete, "new_path": "", "status": "done"}])
                        log_entry.to_csv(log_file_path, mode='a', header=False, index=False)

    # --- Step 4: Clean up old, empty directories ---
    print("\n--- CLEANUP PHASE ---")
    progress_callback(total_files, total_files, "Cleaning up empty folders...")
    for folder in sorted(list(original_folders), key=len, reverse=True):
        if os.path.exists(folder) and not os.listdir(folder) and folder != root_directory:
            os.rmdir(folder)
            print(f"  [CLEAN] Removed empty folder: {folder}")
            log_entry = pd.DataFrame([{"timestamp": timestamp, "action": "rmdir", "original_path": folder, "new_path": "", "status": "done"}])
            log_entry.to_csv(log_file_path, mode='a', header=False, index=False)

    return f"Organization complete. Processed {total_files} files."


def revert_last_sort(root_directory):
    """A true 'undo' that reverts moves, renames, and deletes created folders."""
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    if not os.path.exists(log_file_path): return "Error: Log file not found."
    try:
        print("\n--- STARTING REVERT OPERATION ---")
        df = pd.read_csv(log_file_path)
        last_timestamp = df['timestamp'].iloc[-1]
        session_df = df[df['timestamp'] == last_timestamp]
        
        # Revert moves in reverse order
        for _, row in session_df[session_df['action'] == 'move'].iloc[::-1].iterrows():
            if row['status'] == 'moved':
                try:
                    os.makedirs(os.path.dirname(row['original_path']), exist_ok=True)
                    shutil.move(row['new_path'], row['original_path'])
                    print(f"  [REVERT] Moved back: {os.path.basename(row['original_path'])}")
                except Exception: pass

        # Delete newly created folders
        new_folders = set(os.path.dirname(row['new_path']) for _, row in session_df[session_df['action'] == 'move'].iterrows())
        for folder in sorted(list(new_folders), key=len, reverse=True):
            if os.path.exists(folder) and not os.listdir(folder):
                os.rmdir(folder)
                print(f"  [REVERT] Removed sorted folder: {os.path.basename(folder)}")

        df_remaining = df[df['timestamp'] != last_timestamp]
        if df_remaining.empty: os.remove(log_file_path)
        else: df_remaining.to_csv(log_file_path, index=False)
        return f"Revert complete for session: {last_timestamp}."
    except Exception as e:
        return f"An error occurred during revert: {e}"

def isolate_unprocessable_file(root_directory, file_path):
    # This function is unchanged and still valuable
    try:
        unprocessable_dir = os.path.join(root_directory, "_Unprocessable")
        os.makedirs(unprocessable_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        destination = os.path.join(unprocessable_dir, filename)
        shutil.move(file_path, destination)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not isolate file {file_path}. Error: {e}")