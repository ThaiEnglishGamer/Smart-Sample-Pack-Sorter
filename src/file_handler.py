import os
import shutil
import pandas as pd
from datetime importdatetime

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
        print(f"CRITICAL ERROR: Could not isolate unprocessable file {file_path}. Error: {e}")```

---

### Step 2: The Final `gui.py` to Use the New System

This version is updated to call the new transactional sorting function and pass it the necessary information to keep the UI updated.

**`src/gui.py` (Final Transactional Version)**
```python
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import threading
import multiprocessing
import numpy as np

from file_handler import scan_directory, sort_files_transactional, revert_last_sort, isolate_unprocessable_file
from predictor import SampleSorter, extract_features_live

class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Sample Pack Sorter")
        self.geometry("700x550")
        self.minsize(500, 400)
        self.selected_directory = None
        self.files_to_sort = []
        self.sorter = SampleSorter()
        self.stop_event = None
        self.process_pool = None
        
        # --- (GUI layout code is unchanged) ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.dir_box_label = ctk.CTkLabel(self.main_frame, text="Drag and Drop Your Sample Pack Directory Here", fg_color=("gray75", "gray25"), corner_radius=8)
        self.dir_box_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew", ipady=20)
        self.dir_box_label.drop_target_register(DND_FILES)
        self.dir_box_label.dnd_bind('<<Drop>>', self.handle_drop)
        self.sorting_options_frame = ctk.CTkFrame(self.main_frame)
        self.sorting_options_frame.grid(row=1, column=0, padx=10, pady=10)
        ctk.CTkLabel(self.sorting_options_frame, text="Categorize by:").pack(side="left", padx=(0, 10))
        self.sort_menu = ctk.CTkOptionMenu(self.sorting_options_frame, values=["Sort by Type"])
        self.sort_menu.pack(side="left")
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(self.status_frame, text="Welcome! Please drop a folder to begin.")
        self.status_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=3, column=0, padx=10, pady=20, sticky="s")
        self.start_button = ctk.CTkButton(self.button_frame, text="Start Sorting", state="disabled", command=self.start_sorting_thread)
        self.start_button.pack(side="left", padx=5)
        self.revert_button = ctk.CTkButton(self.button_frame, text="Revert Last Sort", command=self.revert_sorting_thread)
        self.revert_button.pack_forget()
        self.stop_button = ctk.CTkButton(self.button_frame, text="Stop Sorting", command=self.stop_sorting, fg_color="red", hover_color="darkred")
        self.stop_button.pack_forget()

    # --- (handle_drop, set_ui_state, etc. are largely the same) ---
    def handle_drop(self, event):
        path = event.data.strip('{}')
        if os.path.isdir(path):
            self.selected_directory = path
            scan_info = scan_directory(self.selected_directory)
            self.files_to_sort = scan_info["audio_files"]
            log_file_found = scan_info["log_file_found"]
            self.dir_box_label.configure(text=f"Selected: {path}\nFound {len(self.files_to_sort)} audio files.")
            self.status_label.configure(text=f"Ready to process {len(self.files_to_sort)} files.")
            self.progress_bar.set(0)
            if self.files_to_sort: self.start_button.configure(state="normal")
            else: self.start_button.configure(state="disabled")
            if log_file_found: self.revert_button.pack(side="left", padx=5)
            else: self.revert_button.pack_forget()

    def set_ui_state_for_sorting(self, is_sorting):
        if is_sorting:
            self.start_button.pack_forget()
            self.revert_button.pack_forget()
            self.stop_button.pack(side="left", padx=5)
        else:
            self.stop_button.pack_forget()
            self.start_button.pack(side="left", padx=5)
            self.handle_drop_refresh()

    def start_sorting_thread(self):
        self.stop_event = threading.Event()
        self.set_ui_state_for_sorting(is_sorting=True)
        threading.Thread(target=self.run_sorting, daemon=True).start()
    
    def stop_sorting(self):
        if self.stop_event: self.stop_event.set()
        if self.process_pool: self.process_pool.terminate()
        self.status_label.configure(text="Stopping process... please wait.")
    
    def update_progress(self, current, total, message):
        """A dedicated callback function to safely update the UI."""
        progress = current / total
        status_text = f"[{current}/{total}] {message}"
        self.progress_bar.set(progress)
        self.status_label.configure(text=status_text)
    
    def run_sorting(self):
        if not self.sorter.load_model():
            self.after(0, lambda: self.status_label.configure(text="Error: AI model not found."))
            self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
            return

        predictions = []
        total_files = len(self.files_to_sort)
        
        self.process_pool = multiprocessing.Pool()
        try:
            async_results = []
            for file_path in self.files_to_sort:
                if self.stop_event.is_set(): break
                filename_lower = os.path.basename(file_path).lower()
                keyword_category = next((cat for key, cat in self.sorter.keyword_map.items() if key in filename_lower), None)
                
                if keyword_category:
                    async_results.append({'file_path': file_path, 'result': keyword_category, 'is_keyword': True})
                else:
                    result_obj = self.process_pool.apply_async(extract_features_live, (file_path,))
                    async_results.append({'file_path': file_path, 'result': result_obj, 'is_keyword': False})

            for i, item in enumerate(async_results):
                if self.stop_event.is_set(): break
                
                file_path = item['file_path']
                self.after(0, lambda c=i+1, t=total_files, m=f"Classifying: {os.path.basename(file_path)}": self.update_progress(c, t, m))
                
                category = ""
                if item['is_keyword']:
                    category = item['result']
                else:
                    try:
                        features = item['result'].get(timeout=120)
                        if features is not None:
                            pred = self.sorter.model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
                            conf = np.max(pred)
                            category = self.sorter.classes[np.argmax(pred)] if conf >= self.sorter.confidence_threshold else "_Uncategorized"
                        else: category = "_Uncategorized"
                    except multiprocessing.TimeoutError:
                        self.after(0, lambda fp=file_path: isolate_unprocessable_file(self.selected_directory, fp))
                        continue
                    except Exception as e:
                        category = "_Uncategorized"
                
                predictions.append((file_path, category))

            self.process_pool.close()
            self.process_pool.join()

            if not self.stop_event.is_set():
                # --- All predictions are done, now call the transactional sort function ---
                result_message = sort_files_transactional(self.selected_directory, predictions,
                                                          lambda c, t, m: self.after(0, self.update_progress, c, t, m),
                                                          self.stop_event)
                self.after(0, lambda: self.status_label.configure(text=result_message))
        
        finally:
            self.process_pool = None
            self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))

    # --- (Revert and other helper methods are unchanged) ---
    def revert_sorting_thread(self):
        self.set_ui_state_for_sorting(is_sorting=True) # Visually disable buttons
        threading.Thread(target=self.run_revert, daemon=True).start()
    def run_revert(self):
        self.status_label.configure(text="Reverting last sort operation...")
        result_message = revert_last_sort(self.selected_directory)
        self.status_label.configure(text=result_message)
        self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
    def handle_drop_refresh(self):
        self.handle_drop(type('event', (), {'data': self.selected_directory})())