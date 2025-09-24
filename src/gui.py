import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
import time # Import time for UI updates

from file_handler import scan_directory
from audio_classifier import SampleSorter # <-- Import our new Sorter class

class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Sample Pack Sorter")
        self.geometry("700x550") # Increased height for new widgets
        self.minsize(500, 400)

        self.selected_directory = None
        self.files_to_sort = []
        self.sorter = SampleSorter() # Create an instance of the sorter

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1) # Allow row 2 to expand

        self.dir_box_label = ctk.CTkLabel(
            self.main_frame, text="Drag and Drop Your Sample Pack Directory Here",
            fg_color=("gray75", "gray25"), corner_radius=8
        )
        self.dir_box_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew", ipady=20)
        self.dir_box_label.drop_target_register(DND_FILES)
        self.dir_box_label.dnd_bind('<<Drop>>', self.handle_drop)

        self.sorting_options_frame = ctk.CTkFrame(self.main_frame)
        self.sorting_options_frame.grid(row=1, column=0, padx=10, pady=10)
        ctk.CTkLabel(self.sorting_options_frame, text="Categorize by:").pack(side="left", padx=(0, 10))
        self.sort_menu = ctk.CTkOptionMenu(self.sorting_options_frame, values=["Sort by Type"])
        self.sort_menu.pack(side="left")

        # --- NEW: Progress Bar and Status Label ---
        self.status_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.status_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Welcome! Please drop a folder to begin.")
        self.status_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        # --- Buttons ---
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=3, column=0, padx=10, pady=20, sticky="s")
        self.start_button = ctk.CTkButton(self.button_frame, text="Start Sorting", state="disabled", command=self.start_sorting)
        self.start_button.pack(side="left", padx=5)
        self.revert_button = ctk.CTkButton(self.button_frame, text="Revert Last Sort", command=self.revert_sorting)
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
            if self.files_to_sort:
                self.start_button.configure(state="normal")
            else:
                self.start_button.configure(state="disabled")
            if log_file_found:
                self.revert_button.pack(side="left", padx=5)
            else:
                self.revert_button.pack_forget()
        else:
            self.dir_box_label.configure(text="Invalid: Please drop a valid directory")
            # ... (reset state)

    def start_sorting(self):
        # Disable buttons to prevent multiple clicks
        self.start_button.configure(state="disabled")
        self.revert_button.configure(state="disabled")
        self.progress_bar.set(0)

        # Try to load the AI model.
        if not self.sorter.load_model():
            self.status_label.configure(text="Error: AI model not found. Please train first.")
            self.start_button.configure(state="normal") # Re-enable
            return
            
        total_files = len(self.files_to_sort)
        # --- This is where the actual work will happen ---
        # For now, we will just simulate the process and print predictions.
        for i, file_path in enumerate(self.files_to_sort):
            # 1. Get Prediction
            category, reason = self.sorter.predict_category(file_path)
            
            # 2. Update GUI
            progress = (i + 1) / total_files
            self.progress_bar.set(progress)
            self.status_label.configure(text=f"[{i+1}/{total_files}] Classifying: {os.path.basename(file_path)}")
            self.update_idletasks() # Force GUI to refresh

            # 3. Print result to terminal (File moving will be the next step)
            print(f"File: {os.path.basename(file_path):<40} -> Category: {category:<25} (Reason: {reason})")
            
            # time.sleep(0.05) # Optional: slow down for visual effect

        self.status_label.configure(text=f"Classification complete! {total_files} files analyzed.")
        # Re-enable start button, but not revert, as that's a separate action
        self.start_button.configure(state="normal")

    def revert_sorting(self):
        print(f"Reverting changes in directory: {self.selected_directory}")

if __name__ == '__main__':
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()