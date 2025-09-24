import customtkinter as ctk
from gui import App
import multiprocessing # Import the library

def main():
    """Main entry point for the AI Sample Pack Sorter application."""
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()

if __name__ == "__main__":
    # Add this line to ensure multiprocessing works correctly when bundled.
    multiprocessing.freeze_support()
    main()