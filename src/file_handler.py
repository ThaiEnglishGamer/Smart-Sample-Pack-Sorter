import os
import shutil
import pandas as pd
from datetime import datetime

# --- Configuration ---
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg'}
FORBIDDEN_PATH_FRAGMENTS = {'/program files/', '/windows/', '/system/', '/download/', '$recycle.bin'}
MAX_SCAN_DEPTH = 7
LOG_FILE_NAME = "sample_sorter_log.csv"
SORTED_FOLDER_NAME = "_Sorted Samples" # The name of the main folder to put sorted files into.

def scan_directory(root_directory):
    """Scans a directory for audio files."""
    audio_files, log_file_found = [], False
    root_directory = os.path.normpath(root_directory)
    root_depth = root_directory.count(os.sep)

    print(f"Starting scan in: {root_directory}")
    for current_dir, subdirs, files in os.walk(root_directory, topdown=True):
        current_depth = current_dir.count(os.sep)
        if current_depth - root_depth >= MAX_SCAN_DEPTH:
            subdirs[:] = []
            continue
        normalized_current_dir = os.path.normpath(current_dir).lower()
        if any(frag in normalized_current_dir for frag in FORBIDDEN_PATH_FRAGMENTS):
            subdirs[:] = []
            continue
        for filename in files:
            # Look for the log file in the top-level directory only
            if current_dir == root_directory and filename == LOG_FILE_NAME:
                log_file_found = True
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                full_path = os.path.join(current_dir, filename)
                audio_files.append(full_path)
    print(f"Scan complete. Found {len(audio_files)} audio files.")
    return {"audio_files": audio_files, "log_file_found": log_file_found}

# --- NEW: Function to perform the sorting ---
def sort_files(root_directory, predictions):
    """
    Moves files to their new sorted locations and creates a CSV log.

    Args:
        root_directory (str): The main directory being sorted.
        predictions (list of tuples): A list where each item is
                                      (original_path, predicted_category).
    
    Returns:
        str: A message summarizing the result.
    """
    log_data = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the main destination folder
    main_sorted_dir = os.path.join(root_directory, SORTED_FOLDER_NAME)
    os.makedirs(main_sorted_dir, exist_ok=True)

    for original_path, category in predictions:
        try:
            # Create the category sub-folder if it doesn't exist
            category_dir = os.path.join(main_sorted_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            filename = os.path.basename(original_path)
            new_path = os.path.join(category_dir, filename)
            
            # Move the file
            shutil.move(original_path, new_path)
            
            # Record the successful move for the log
            log_data.append({
                "timestamp": timestamp,
                "original_path": original_path,
                "new_path": new_path,
                "status": "moved"
            })
        except Exception as e:
            print(f"Could not move file {original_path}. Error: {e}")
            log_data.append({
                "timestamp": timestamp,
                "original_path": original_path,
                "new_path": "N/A",
                "status": f"error: {e}"
            })

    # Create or append to the log file using pandas
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    log_df = pd.DataFrame(log_data)
    
    if os.path.exists(log_file_path):
        log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file_path, mode='w', header=True, index=False)

    return f"Sort complete. Processed {len(predictions)} files."

# --- NEW: Function to revert the last sort operation ---
def revert_last_sort(root_directory):
    """
    Reads the log file and moves files back to their original locations.
    """
    log_file_path = os.path.join(root_directory, LOG_FILE_NAME)
    if not os.path.exists(log_file_path):
        return "Error: Log file not found. Cannot revert."

    try:
        df = pd.read_csv(log_file_path)
        # Find the timestamp of the most recent sort operation
        last_timestamp = df['timestamp'].iloc[-1]
        last_sort_df = df[df['timestamp'] == last_timestamp]

        files_to_revert = last_sort_df[last_sort_df['status'] == 'moved']

        for index, row in files_to_revert.iterrows():
            try:
                # Ensure the original directory exists before moving back
                original_dir = os.path.dirname(row['original_path'])
                os.makedirs(original_dir, exist_ok=True)
                
                # Move the file back
                shutil.move(row['new_path'], row['original_path'])
            except FileNotFoundError:
                print(f"Warning: Could not find file to revert: {row['new_path']}. It may have been moved or deleted.")
            except Exception as e:
                print(f"Could not revert {row['new_path']}. Error: {e}")

        # Remove the reverted entries from the log
        df_remaining = df[df['timestamp'] != last_timestamp]
        if df_remaining.empty:
            os.remove(log_file_path) # Delete the log if it's now empty
        else:
            df_remaining.to_csv(log_file_path, index=False)
        
        return f"Revert complete for sort operation from {last_timestamp}."

    except Exception as e:
        return f"An error occurred during revert: {e}"