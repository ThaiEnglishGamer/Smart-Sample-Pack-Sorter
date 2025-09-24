import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import threading

# --- MODIFIED: Import the new functions ---
from file_handler import scan_directory, sort_files, revert_last_sort
from audio_classifier import SampleSorter

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
        # --- MODIFIED: The model needs to be loaded from the new 'models' directory ---
        self.sorter = SampleSorter()

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        # ... (rest of __init__ is the same until the handler/sorter methods)
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

    def set_ui_state(self, is_busy):
        """Helper function to enable/disable UI elements."""
        state = "disabled" if is_busy else "normal"
        self.start_button.configure(state=state)
        # Only re-enable revert if a log file exists
        if not is_busy and os.path.exists(os.path.join(self.selected_directory, "sample_sorter_log.csv")):
            self.revert_button.configure(state="normal")
        else:
             self.revert_button.configure(state="disabled")

    # --- MODIFIED: Sorting is now run in a separate thread to prevent GUI freezing ---
    def start_sorting_thread(self):
        self.set_ui_state(is_busy=True)
        threading.Thread(target=self.run_sorting, daemon=True).start()
    
    def revert_sorting_thread(self):
        self.set_ui_state(is_busy=True)
        threading.Thread(target=self.run_revert, daemon=True).start()

    def run_sorting(self):
        """The actual sorting logic that runs in a thread."""
        self.progress_bar.set(0)
        
        # --- MODIFIED: Load model from the correct directory ---
        # We need to temporarily change directory so the relative paths in audio_classifier.py work
        os.chdir('..') # Go up to the main project directory
        model_loaded = self.sorter.load_model()
        os.chdir('src') # Go back to the src directory

        if not model_loaded:
            self.status_label.configure(text="Error: AI model not found in 'models' folder.")
            self.set_ui_state(is_busy=False)
            return
            
        total_files = len(self.files_to_sort)
        predictions = []
        for i, file_path in enumerate(self.files_to_sort):
            category, reason = self.sorter.predict_category(file_path)
            predictions.append((file_path, category))
            
            # --- MODIFIED: Use `after` to safely update GUI from a thread ---
            progress = (i + 1) / total_files
            status_text = f"[{i+1}/{total_files}] Classifying: {os.path.basename(file_path)}"
            self.after(0, lambda p=progress, s=status_text: self.update_ui(p, s))

        # Now, call the file handler to move the files
        self.after(0, lambda: self.status_label.configure(text="All files classified. Now moving..."))
        result_message = sort_files(self.selected_directory, predictions)
        self.after(0, lambda: self.status_label.configure(text=result_message))
        
        # Rescan the directory to update the file count and revert button status
        self.after(0, lambda: self.handle_drop_refresh())
        self.after(0, lambda: self.set_ui_state(is_busy=False))

    def run_revert(self):
        """The actual reverting logic that runs in a thread."""
        self.status_label.configure(text="Reverting last sort operation...")
        result_message = revert_last_sort(self.selected_directory)
        self.status_label.configure(text=result_message)
        # Rescan the directory to update UI
        self.after(0, lambda: self.handle_drop_refresh())
        self.after(0, lambda: self.set_ui_state(is_busy=False))

    def update_ui(self, progress, status_text):
        """Safely updates UI elements."""
        self.progress_bar.set(progress)
        self.status_label.configure(text=status_text)
    
    def handle_drop_refresh(self):
        """A version of handle_drop that doesn't need an event, for refreshing."""
        self.handle_drop(type('event', (), {'data': self.selected_directory})())