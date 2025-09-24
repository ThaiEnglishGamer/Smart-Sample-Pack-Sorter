import os
import multiprocessing

# --- NEW: Intelligent Hardware Detection ---
# This block MUST run before any other part of the application.

# Step 1: Suppress TensorFlow's non-critical log messages (like the CUDA init error).
# '2' means only print errors, hiding warnings and info.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import customtkinter as ctk
from gui import App

def configure_hardware():
    """
    Checks for a compatible GPU and configures TensorFlow accordingly.
    """
    # Step 2: Ask TensorFlow to list the available GPUs.
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        # If the list is not empty, we have a GPU.
        print(f"INFO: Found {len(gpus)} compatible CUDA GPU(s). The application will use the GPU.")
        
        # Step 3 (Best Practice): Configure the GPU to use memory on-demand.
        # This prevents TensorFlow from allocating all GPU memory at once.
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # This might happen if the memory growth option is set too late.
            # It's not a fatal error.
            print(f"Warning: Could not set memory growth for GPU. Error: {e}")
    else:
        # If the list is empty, we are on a CPU.
        print("INFO: No compatible CUDA GPU found. The application will run on the CPU.")

def main():
    """
    Main entry point for the AI Sample Pack Sorter application.
    """
    # Step 4: Run our hardware configuration function first.
    configure_hardware()
    
    # Now, launch the GUI as before.
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = App()
    app.mainloop()

if __name__ == "__main__":
    # This ensures multiprocessing works correctly when bundled into an executable.
    multiprocessing.freeze_support()
    main()