import os

# --- Configuration: Easy to change later ---

# Define which file extensions are considered audio files.
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg'}

# Define parts of paths that should be ignored to avoid scanning system folders.
FORBIDDEN_PATH_FRAGMENTS = {'/program files/', '/windows/', '/system/', '/download/', '$recycle.bin'}

# The maximum number of folders deep the scanner will go.
MAX_SCAN_DEPTH = 7

# The name of our log file that enables the "revert" feature.
LOG_FILE_NAME = "sample_sorter_log.csv"


def scan_directory(root_directory):
    """
    Scans a directory for audio files based on the rules defined above.

    Args:
        root_directory (str): The absolute path to the folder to scan.

    Returns:
        dict: A dictionary containing 'audio_files' (a list of paths)
              and 'log_file_found' (a boolean).
    """
    audio_files = []
    log_file_found = False
    
    # Normalize the path for consistent separator usage
    root_directory = os.path.normpath(root_directory)
    root_depth = root_directory.count(os.sep)

    print(f"Starting scan in: {root_directory}")

    for current_dir, subdirs, files in os.walk(root_directory, topdown=True):
        # --- Rule 1: Max Depth ---
        current_depth = current_dir.count(os.sep)
        if current_depth - root_depth >= MAX_SCAN_DEPTH:
            # By clearing the subdirs list, we tell os.walk() not to go any deeper.
            subdirs[:] = []
            continue

        # --- Rule 2: Forbidden Directories ---
        # We check a normalized, lowercased version of the path for fragments.
        normalized_current_dir = os.path.normpath(current_dir).lower()
        if any(frag in normalized_current_dir for frag in FORBIDDEN_PATH_FRAGMENTS):
            # Skip this directory and all its subdirectories
            subdirs[:] = []
            print(f"Skipping forbidden directory: {current_dir}")
            continue
            
        # --- Process Files in the Current Directory ---
        for filename in files:
            # Check for our log file
            if filename == LOG_FILE_NAME:
                log_file_found = True
            
            # Check for supported audio extensions
            # We check the lowercased extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                full_path = os.path.join(current_dir, filename)
                audio_files.append(full_path)

    print(f"Scan complete. Found {len(audio_files)} audio files.")
    if log_file_found:
        print(f"Found an existing log file: {LOG_FILE_NAME}")

    return {
        "audio_files": audio_files,
        "log_file_found": log_file_found
    }