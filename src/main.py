import os
import multiprocessing

# --- Intelligent Hardware Configuration ---
# This block runs first to set up TensorFlow correctly.

# Step 1: Set the log level to hide non-critical messages.
# '2' means only print errors, hiding warnings and info about CPU/CUDA.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import customtkinter as ctk
from gui import App

def configure_hardware():
    """
    Checks for a compatible GPU. If found, configures it. If not,
    silently falls back to CPU without showing errors.
    """
    try:
        # Step 2: Ask TensorFlow to list the available GPUs.
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # If the list is not empty, we have a GPU.
            print(f"INFO: Found {len(gpus)} compatible CUDA GPU(s). The application will use the GPU.")
            # Configure the GPU to use memory on-demand (a best practice).
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            # If the list is empty, we are on a CPU.
            print("INFO: No compatible CUDA GPU found. The application will run on the CPU.")
    except Exception as e:
        # If any error occurs during GPU setup, just print a warning and proceed with CPU.
        print(f"WARNING: Could not configure GPU. Falling back to CPU. Error: {e}")

def main():
    """Main entry point for the AI Sample Pack Sorter application."""
    # Step 3: Run our hardware configuration function first.
    configure_hardware()
    
    # Now, launch the GUI.
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = App()
    app.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()