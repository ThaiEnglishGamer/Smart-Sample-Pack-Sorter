import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os

# --- NEW: Import our file handler function ---
from file_handler import scan_directory

class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

class App(Tk):
    def __init__(self):
        super().__init__()

        self.title("AI Sample Pack Sorter")
        self.geometry("700x500")
        self.minsize(500, 350)

        # --- NEW: Add properties to store scan results ---
        self.selected_directory = None
        self.files_to_sort = []

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.dir_box_label = ctk.CTkLabel(
            self.main_frame,
            text="Drag and Drop Your Sample Pack Directory Here",
            fg_color=("gray75", "gray25"),
            corner_radius=8
        )
        self.dir_box_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew", ipady=20)

        self.dir_box_label.drop_target_register(DND_FILES)
        self.dir_box_label.dnd_bind('<<Drop>>', self.handle_drop)

        self.sorting_options_frame = ctk.CTkFrame(self.main_frame)
        self.sorting_options_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        self.sort_label = ctk.CTkLabel(self.sorting_options_frame, text="Categorize by:")
        self.sort_label.pack(side="left", padx=(0, 10))

        self.sort_menu = ctk.CTkOptionMenu(
            self.sorting_options_frame,
            values=["Sort by Type"]
        )
        self.sort_menu.pack(side="left")

        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=2, column=0, padx=10, pady=20, sticky="s")

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Sorting",
            state="disabled",
            command=self.start_sorting
        )
        self.start_button.pack(side="left", padx=5)

        self.revert_button = ctk.CTkButton(
            self.button_frame,
            text="Revert Last Sort",
            command=self.revert_sorting
        )
        self.revert_button.pack_forget()

    def handle_drop(self, event):
        """
        Handles the dropping of a folder, runs the scanner, and updates the GUI.
        """
        path = event.data.strip('{}')
        if os.path.isdir(path):
            self.selected_directory = path
            
            # --- MODIFIED: Run the scan and process results ---
            scan_info = scan_directory(self.selected_directory)
            self.files_to_sort = scan_info["audio_files"]
            log_file_found = scan_info["log_file_found"]

            # Update label with feedback
            self.dir_box_label.configure(text=f"Selected: {path}\nFound {len(self.files_to_sort)} audio files.")
            
            # Enable start button if files were found
            if self.files_to_sort:
                self.start_button.configure(state="normal")
            else:
                self.start_button.configure(state="disabled")

            # Show or hide the revert button
            if log_file_found:
                self.revert_button.pack(side="left", padx=5)
            else:
                self.revert_button.pack_forget()
        else:
            self.dir_box_label.configure(text="Invalid: Please drop a valid directory")
            self.selected_directory = None
            self.files_to_sort = []
            self.start_button.configure(state="disabled")
            self.revert_button.pack_forget()

    def start_sorting(self):
        """Placeholder for the sorting logic."""
        print(f"Starting to sort {len(self.files_to_sort)} files in directory: {self.selected_directory}")
        print(f"Sorting method: {self.sort_menu.get()}")
        # We will soon replace this with the actual AI classification and file moving.

    def revert_sorting(self):
        """Placeholder for the reverting logic."""
        print(f"Reverting changes in directory: {self.selected_directory}")


if __name__ == '__main__':
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()