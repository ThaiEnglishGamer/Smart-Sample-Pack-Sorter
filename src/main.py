import customtkinter as ctk
from gui import App  # Import the App class from our gui.py file

def main():
    """
    Main entry point for the AI Sample Pack Sorter application.
    Initializes and runs the GUI.
    """
    # Set the appearance and theme for the application
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    # Create an instance of our main application window
    app = App()
    
    # Run the application's main loop
    app.mainloop()

if __name__ == "__main__":
    # This ensures the main function is called only when this script is executed
    main()