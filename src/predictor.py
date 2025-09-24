import os
import subprocess
import numpy as np
import librosa
import tensorflow as tf

# --- Configuration ---
MODEL_DIR = os.path.join('..', 'models')
MODEL_FILE_NAME = os.path.join(MODEL_DIR, "audio_classifier_model.keras")
CLASSES_FILE_NAME = os.path.join(MODEL_DIR, "classes.npy")
AUDIO_TARGET_LENGTH = 2
TARGET_SR = 44100

def extract_features_live(file_path, n_mfcc=13):
    """
    This is the robust ffmpeg extractor for real-time prediction by the GUI.
    """
    try:
        command = [
            'ffmpeg', '-i', file_path, '-f', 's16le', '-ac', '1',
            '-ar', str(TARGET_SR), '-loglevel', 'error', '-'
        ]
        proc = subprocess.run(command, capture_output=True, check=True)
        raw_audio = proc.stdout
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        target_samples = AUDIO_TARGET_LENGTH * TARGET_SR
        audio_np = librosa.util.fix_length(data=audio_np, size=target_samples)
        mfccs = librosa.feature.mfcc(y=audio_np, sr=TARGET_SR, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Prediction feature extraction failed for {os.path.basename(file_path)}: {e}")
        return None

class SampleSorter:
    def __init__(self, confidence_threshold=0.60):
        self.model, self.classes, self.confidence_threshold = None, None, confidence_threshold
        self.keyword_map = {
            'kick': 'Drums_Kick', 'snare': 'Drums_Snare', 'clap': 'Drums_Clap', 'hat': 'Drums_Cymbals',
            'hihat': 'Drums_Cymbals', 'cymbal': 'Drums_Cymbals', 'ride': 'Drums_Cymbals', 'crash': 'Drums_Crash',
            'fill': 'Drums_Fills', 'perc': 'Drums_Percussion', 'drumloop': 'Loops_Drums', 'drum loop': 'Loops_Drums',
            'bassline': 'Loops_Synth_Bass', 'bass loop': 'Loops_Synth_Bass', 'chord loop': 'Loops_Synth_Chords',
            'lead loop': 'Loops_Synth_Lead', 'melody': 'Loops_Synth_Lead', 'piano': 'Sample_Piano',
            'synth': 'Sample_Synth', 'bass': 'Sample_Synth_Bass', 'shot': 'Sample_Oneshot', 'impact': 'FX_Impact',
            'riser': 'FX_Riser', 'rise': 'FX_Riser', 'sweep': 'FX_Sweep', 'noise': 'FX_Sweep',
            'ambient': 'FX_Ambient', 'pad': 'FX_Ambient', 'vocal': 'FX_Vocals', 'vox': 'FX_Vocals'
        }

    def load_model(self):
        """Loads the trained Keras model and class names from the models directory."""
        if os.path.exists(MODEL_FILE_NAME) and os.path.exists(CLASSES_FILE_NAME):
            self.model = tf.keras.models.load_model(MODEL_FILE_NAME)
            self.classes = np.load(CLASSES_FILE_NAME, allow_pickle=True)
            print("AI model and classes loaded successfully for prediction.")
            return True
        else:
            print("Model files not found in 'models' directory.")
            return False

    def predict_category(self, file_path):
        """Predicts category using the hybrid Filename + AI approach."""
        # --- Stage 1: Filename Analysis ---
        filename = os.path.basename(file_path).lower()
        for keyword, category in self.keyword_map.items():
            if keyword in filename:
                return category, "Keyword"

        # --- Stage 2: AI Audio Analysis (if no keyword found) ---
        if not self.model:
            return "_Uncategorized", "AI Model not loaded"

        features = extract_features_live(file_path) # Use the ffmpeg extractor
        if features is None:
            return "_Uncategorized", "Feature Extraction Error"

        features = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features, verbose=0)[0]
        confidence = np.max(prediction)

        if confidence >= self.confidence_threshold:
            predicted_index = np.argmax(prediction)
            return self.classes[predicted_index], f"AI ({confidence*100:.0f}%)"
        else:
            return "_Uncategorized", f"AI Low Confidence ({confidence*100:.0f}%)"