import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import threading
import subprocess
import json
import sys
import numpy as np

from file_handler import scan_directory, sort_files_transactional, revert_last_sort, isolate_unprocessable_file
from predictor import SampleSorter

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
        self.stop_event = threading.Event()
        self.worker_process = None

        # --- (GUI layout is unchanged) ---
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
        self.stop_event.clear()
        self.set_ui_state_for_sorting(is_sorting=True)
        threading.Thread(target=self.run_sorting, daemon=True).start()
    
    def stop_sorting(self):
        self.status_label.configure(text="Stopping... please wait.")
        self.stop_event.set()
        if self.worker_process:
            self.worker_process.terminate()

    def run_sorting(self):
        if not self.sorter.load_model():
            self.after(0, lambda: self.status_label.configure(text="Error: AI model not found."))
            self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
            return
            
        predictions = []
        files_for_ai = []

        # --- STAGE 1: Fast Keyword Classification ---
        self.after(0, self.update_progress, 0, 1, "Phase 1: Classifying files by name...")
        for file_path in self.files_to_sort:
            filename_lower = os.path.basename(file_path).lower()
            keyword_category = next((cat for key, cat in self.sorter.keyword_map.items() if key in filename_lower), None)
            if keyword_category:
                predictions.append((file_path, keyword_category))
            else:
                files_for_ai.append(file_path)
        
        # --- STAGE 2: AI Classification in an Isolated Process ---
        if files_for_ai:
            self.after(0, self.update_progress, 0, 1, f"Phase 2: Analyzing {len(files_for_ai)} audio files with AI...")
            try:
                # Get the path to the python executable running this script
                python_executable = sys.executable
                worker_script = os.path.join(os.path.dirname(__file__), 'feature_extractor_worker.py')
                command = [python_executable, worker_script] + files_for_ai
                
                self.worker_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = self.worker_process.communicate()
                self.worker_process = None # Clear process after it's done

                if self.stop_event.is_set():
                    self.after(0, self.status_label.configure, {"text": "Sorting aborted by user."})
                    self.after(0, self.set_ui_state_for_sorting, False)
                    return

                if stderr:
                    print("--- Worker Process Errors ---")
                    print(stderr)

                ai_results = json.loads(stdout)
                
                # Process successful results
                for file_path, features_list in ai_results["success"].items():
                    features = np.array(features_list)
                    pred = self.sorter.model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
                    conf = np.max(pred)
                    category = self.sorter.classes[np.argmax(pred)] if conf >= self.sorter.confidence_threshold else "_Uncategorized"
                    predictions.append((file_path, category))
                
                # Handle failed results
                for file_path in ai_results["failed"]:
                    self.after(0, lambda fp=file_path: isolate_unprocessable_file(self.selected_directory, fp))

            except Exception as e:
                self.after(0, self.status_label.configure, {"text": f"A critical error occurred: {e}"})
                self.after(0, self.set_ui_state_for_sorting, False)
                return

        # --- STAGE 3: Transactional File Moving ---
        self.after(0, self.status_label.configure, {"text": "Phase 3: Moving files to sorted folders..."})
        result_message = sort_files_transactional(self.selected_directory, predictions,
                                                  lambda c, t, m: self.after(0, self.update_progress, c, t, m),
                                                  self.stop_event)
        self.after(0, self.status_label.configure, {"text": result_message})
        self.after(0, self.set_ui_state_for_sorting, False)

    # ... (the rest of the methods are unchanged)
    def update_progress(self, current, total, message):
        self.progress_bar.set(current / total)
        self.status_label.configure(text=message)
    def revert_sorting_thread(self):
        self.set_ui_state_for_sorting(True)
        threading.Thread(target=self.run_revert, daemon=True).start()
    def run_revert(self):
        self.status_label.configure(text="Reverting last sort operation...")
        result_message = revert_last_sort(self.selected_directory)
        self.status_label.configure(text=result_message)
        self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
    def handle_drop_refresh(self):
        self.handle_drop(type('event', (), {'data': self.selected_directory})())