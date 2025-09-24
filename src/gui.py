import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import threading
import multiprocessing
import numpy as np

from file_handler import scan_directory, sort_files, revert_last_sort
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
        self.stop_event = None # This will be our flag for the stop button

        # --- Main Frame and Widgets ---
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

        # --- Button Frame ---
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=3, column=0, padx=10, pady=20, sticky="s")
        self.start_button = ctk.CTkButton(self.button_frame, text="Start Sorting", state="disabled", command=self.start_sorting_thread)
        self.start_button.pack(side="left", padx=5)
        self.revert_button = ctk.CTkButton(self.button_frame, text="Revert Last Sort", command=self.revert_sorting_thread)
        self.revert_button.pack_forget()
        
        # --- NEW: The Emergency Stop Button ---
        self.stop_button = ctk.CTkButton(self.button_frame, text="Stop Sorting", command=self.stop_sorting, fg_color="red", hover_color="darkred")
        self.stop_button.pack_forget() # Hidden by default


    def handle_drop(self, event):
        # ... (this method is unchanged)
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
        """Manages which buttons are visible and active during the sorting process."""
        if is_sorting:
            self.start_button.pack_forget()
            self.revert_button.pack_forget()
            self.stop_button.pack(side="left", padx=5)
        else:
            self.stop_button.pack_forget()
            self.start_button.pack(side="left", padx=5)
            # After sorting, rescan to see if the revert button should be shown
            self.handle_drop_refresh()

    def start_sorting_thread(self):
        self.stop_event = threading.Event() # Create a new, clear flag
        self.set_ui_state_for_sorting(is_sorting=True)
        threading.Thread(target=self.run_sorting, daemon=True).start()
    
    def stop_sorting(self):
        """Sets the stop flag when the stop button is clicked."""
        if self.stop_event:
            self.status_label.configure(text="Stopping process...")
            self.stop_event.set()

    def run_sorting(self):
        """The main sorting logic, now fully non-blocking and responsive."""
        if not self.sorter.load_model():
            self.after(0, lambda: self.status_label.configure(text="Error: AI model not found in 'models' folder."))
            self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
            return

        total_files = len(self.files_to_sort)
        predictions = []
        
        with multiprocessing.Pool() as pool:
            # --- MODIFIED: Use apply_async for non-blocking processing ---
            async_results = []
            for file_path in self.files_to_sort:
                # Stage 1: Check keywords first (fast)
                filename_lower = os.path.basename(file_path).lower()
                keyword_category = None
                for keyword, category in self.sorter.keyword_map.items():
                    if keyword in filename_lower:
                        keyword_category = category
                        break
                
                # Stage 2: If no keyword, send to background process for feature extraction
                if keyword_category:
                    async_results.append({'file_path': file_path, 'result': keyword_category, 'is_keyword': True})
                else:
                    result_obj = pool.apply_async(extract_features_live, (file_path,))
                    async_results.append({'file_path': file_path, 'result': result_obj, 'is_keyword': False})

            # --- Now, collect results as they become available ---
            for i, item in enumerate(async_results):
                if self.stop_event.is_set():
                    pool.terminate() # Forcefully stop background processes
                    self.after(0, lambda: self.status_label.configure(text="Sorting aborted by user."))
                    self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))
                    return

                file_path = item['file_path']
                category = ""
                
                if item['is_keyword']:
                    category = item['result']
                else: # It's an AI prediction, get the result from the background process
                    features = item['result'].get() # This will wait here until this specific file is done
                    if features is not None:
                        features_expanded = np.expand_dims(features, axis=0)
                        prediction = self.sorter.model.predict(features_expanded, verbose=0)[0]
                        confidence = np.max(prediction)
                        if confidence >= self.sorter.confidence_threshold:
                            category = self.sorter.classes[np.argmax(prediction)]
                        else:
                            category = "_Uncategorized"
                    else:
                        category = "_Uncategorized"
                
                predictions.append((file_path, category))

                # Update UI
                progress = (i + 1) / total_files
                status_text = f"[{i+1}/{total_files}] Classifying: {os.path.basename(file_path)}"
                self.after(0, lambda p=progress, s=status_text: self.update_ui(p, s))

        # --- Proceed to move files only if we didn't stop ---
        if not self.stop_event.is_set():
            self.after(0, lambda: self.status_label.configure(text="All files classified. Now moving..."))
            result_message = sort_files(self.selected_directory, predictions)
            self.after(0, lambda: self.status_label.configure(text=result_message))

        self.after(0, lambda: self.set_ui_state_for_sorting(is_sorting=False))

    def revert_sorting_thread(self):
        # ... (this method is unchanged and works as intended)
        self.start_button.configure(state="disabled")
        self.revert_button.configure(state="disabled")
        threading.Thread(target=self.run_revert, daemon=True).start()

    def run_revert(self):
        # ... (this method is unchanged and works as intended)
        self.status_label.configure(text="Reverting last sort operation...")
        result_message = revert_last_sort(self.selected_directory)
        self.status_label.configure(text=result_message)
        self.after(0, lambda: self.handle_drop_refresh())

    def update_ui(self, progress, status_text):
        # ... (this method is unchanged and works as intended)
        self.progress_bar.set(progress)
        self.status_label.configure(text=status_text)
    
    def handle_drop_refresh(self):
        # ... (this method is unchanged and works as intended)
        self.handle_drop(type('event', (), {'data': self.selected_directory})())