# AI Sample Pack Sorter

An intelligent, AI-powered tool designed to automatically clean, categorize, and organize messy music production sample pack libraries. This utility transforms chaotic folders into a perfectly structured, ready-to-use library for producers and sound designers.



---

## ⚠️ Project Status: In Active Development

Please note that this project is currently a work-in-progress. While the core functionality is in place, it is still undergoing testing and refinement. You may encounter bugs or unexpected behavior. Feel free to report any issues or suggest features!

---

## Key Features

*   **Intelligent Audio Classification:** Utilizes a neural network to "listen" to and identify different types of audio samples (kicks, snares, synths, loops, FX, etc.).
*   **Hybrid Sorting Logic:** Combines filename keyword analysis for speed and accuracy with AI analysis for poorly named files.
*   **Comprehensive File Handling:** Not just for audio! It intelligently sorts VST presets, DAW project files, MIDI, and more into their own dedicated folders.
*   **Automatic Duplicate Management:** Saves disk space by finding and removing perfect duplicates (based on file content, not just name). Safely renames files with name collisions.
*   **Fail-Safe & Transactional:** Creates a detailed log *before* taking action. If the process is interrupted, no data is lost, and it can be safely reverted.
*   **"Perfect Undo" Revert:** A powerful revert function that uses the log to move all files back to their original locations and cleanly remove the newly created sorted folders.
*   **User-Friendly GUI:** A simple drag-and-drop interface with real-time progress updates and a "Stop" button for full control.
*   **Advanced AI Training Workflow:** Includes a powerful, configurable training script that can automatically search for the best-performing AI model overnight.

---

## Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/sample-pack-sorter.git
cd sample-pack-sorter
```

### 2. Install System Dependencies

This application requires **Python 3.10+**, **FFmpeg**, and the **Tkinter** library to be installed on your system.

*   **On Debian / Ubuntu / Mint:**
    ```bash
    sudo apt-get update
    sudo apt-get install python3-tk ffmpeg
    ```
*   **On Arch Linux / Manjaro:**
    ```bash
    sudo pacman -S tk ffmpeg
    ```
*   **On Windows:**
    *   Python and Tkinter are usually included in the official Python installer. Ensure you check the box "Add Python to PATH" during installation.
    *   Install [FFmpeg for Windows](https://www.gyan.dev/ffmpeg/builds/) and add its `bin` folder to your system's PATH.

### 3. Set Up the Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv sample_sorter_env

# Activate it
# On Linux/macOS:
source sample_sorter_env/bin/activate
# On Windows (Command Prompt):
sample_sorter_env\Scripts\activate.bat
```

### 4. Install Python Packages

All required Python packages are listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

## How to Use

The workflow is divided into two main parts: **Training the AI** and **Running the Sorter**.

### Step 1: Prepare Your Training Data

The AI's intelligence depends on the quality and quantity of your training data.

1.  Navigate to the `dataset/` folder.
2.  Inside, you will find sub-folders for each sound category (e.g., `Drums_Kick`, `FX_Riser`, `Sample_Synth`).
3.  Populate these folders with high-quality `.wav` files that match the category. The more samples you add to each folder (aim for 50+), the more accurate your AI will become.

### Step 2: Train the AI Model

Before you can sort, you must train the AI on your custom dataset.

#### If you have changed the dataset (added/removed samples):

You must run a two-stage process. First, pre-process the audio, then train the model.

```bash
# Navigate to the source directory
cd src

# 1. Run the pre-processor (this can be slow)
python preprocess.py

# 2. Run the trainer
python audio_classifier.py
```

#### To experiment and find the best model:

The training script `audio_classifier.py` is a powerful tool.

*   **Manual Mode:** Open the script and edit the `CONTROL PANEL` section at the top to set your desired training parameters. Then run it:
    ```bash
    python audio_classifier.py
    ```
*   **Automatic Random Search Mode:** To have the script automatically experiment overnight to find the best possible model, run it with the `--random-search` flag. It will run forever until you stop it with `Ctrl+C`.
    ```bash
    python audio_classifier.py --random-search
    ```

The best model is automatically saved to the `models/` folder.

### Step 3: Run the Sorter Application

Once your model is trained, you're ready to organize!

```bash
# Navigate to the source directory
cd src

# Run the main application
python main.py
```

A GUI window will appear. Simply drag and drop the messy sample pack folder you want to organize onto the window, click "Start Sorting," and let the program do its magic.

---

## The Sorting Logic Explained

The program organizes files in a specific, intelligent order:

1.  **Audio Keyword Search:** First, it quickly sorts any audio file with obvious keywords in its name (e.g., "kick," "snare," "loop").
2.  **AI Analysis:** Any remaining audio files are analyzed by the trained neural network and sorted into their predicted category.
3.  **Non-Audio Categorization:** All other files (presets, project files, MIDI, etc.) are sorted into dedicated folders based on their file extension.
4.  **Duplicate Handling:** After all files are moved, the program scans for duplicates. Perfect duplicates are removed to save space, while name collisions are safely renamed.
5.  **Cleanup:** Finally, all the old, now-empty subfolders are deleted, leaving a perfectly clean and organized library.

---

## Contributing

Contributions are welcome! If you have ideas for new features, find a bug, or want to improve the code, please feel free to open an issue or submit a pull request.
