import os
import numpy as np
import librosa
import subprocess # NEW: To call ffmpeg
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
MODEL_FILE_NAME = "audio_classifier_model.keras"
CLASSES_FILE_NAME = "classes.npy"
AUDIO_TARGET_LENGTH = 2
MIN_SAMPLES_PER_CLASS = 10
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg')
# --- NEW: Define a standard sample rate for processing ---
TARGET_SR = 44100

# ==============================================================================
#  CLASS 1: THE SORTER (This class is fine and does not need changes)
# ==============================================================================
class SampleSorter:
    # This class remains the same, but it will now call the new, robust extract_features
    # in the AudioClassifier class. For simplicity, we'll embed the new logic here as well.
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
        if os.path.exists(MODEL_FILE_NAME) and os.path.exists(CLASSES_FILE_NAME):
            self.model = tf.keras.models.load_model(MODEL_FILE_NAME)
            self.classes = np.load(CLASSES_FILE_NAME, allow_pickle=True)
            return True
        return False
    def predict_category(self, file_path):
        filename = os.path.basename(file_path).lower()
        for keyword, category in self.keyword_map.items():
            if keyword in filename: return category, "Keyword"
        if not self.model: return "_Uncategorized", "AI Model not loaded"
        # Use the master feature extractor from the classifier
        features = AudioClassifier().extract_features(file_path, is_training=False)
        if features is None: return "_Uncategorized", "Feature Extraction Error"
        features = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features, verbose=0)[0]
        confidence = np.max(prediction)
        if confidence >= self.confidence_threshold:
            return self.classes[np.argmax(prediction)], f"AI ({confidence*100:.0f}%)"
        else:
            return "_Uncategorized", f"AI Low Confidence ({confidence*100:.0f}%)"


# ==============================================================================
#  CLASS 2: THE TRAINER
# ==============================================================================
class AudioClassifier:
    def __init__(self):
        self.model, self.encoder = None, LabelEncoder()

    # --- COMPLETELY REWRITTEN to use FFmpeg ---
    def extract_features(self, file_path, n_mfcc=13, is_training=True):
        """
        Uses FFmpeg to load and standardize audio, then extracts features.
        This is the most robust method and avoids librosa directly touching problem files.
        """
        try:
            # Build the FFmpeg command
            command = [
                'ffmpeg',
                '-i', file_path,      # Input file
                '-f', 's16le',        # Format: signed 16-bit little-endian PCM
                '-ac', '1',           # Audio Channels: 1 (mono)
                '-ar', str(TARGET_SR),# Audio Rate: 44100 Hz
                '-loglevel', 'error', # Don't print ffmpeg info, only errors
                '-'                   # Output to stdout
            ]
            
            # Run FFmpeg and get the raw audio data from stdout
            proc = subprocess.run(command, capture_output=True, check=True)
            raw_audio = proc.stdout
            
            # Convert the raw byte data to a numpy array that librosa can understand
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            
            # Normalize the audio data (ffmpeg gives integers, librosa expects floats between -1 and 1)
            audio_np /= 32768.0

            # Now that we have clean, standardized audio, use librosa for feature extraction
            target_samples = AUDIO_TARGET_LENGTH * TARGET_SR
            audio_np = librosa.util.fix_length(data=audio_np, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio_np, sr=TARGET_SR, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)

        except subprocess.CalledProcessError as e:
            # This catches errors if ffmpeg fails to convert the file
            if is_training:
                print(f"      -> SKIPPING: FFmpeg could not process file: {os.path.basename(file_path)}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            if is_training:
                print(f"      -> SKIPPING: An unexpected error occurred with {os.path.basename(file_path)} | Error: {e}")
            return None

    def load_data(self, dataset_path):
        features, labels = [], []
        
        all_class_labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        valid_class_labels = []
        print("\n--- Analyzing Dataset ---")
        for label in all_class_labels:
            class_dir = os.path.join(dataset_path, label)
            sample_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)]
            num_samples = len(sample_files)
            if num_samples < MIN_SAMPLES_PER_CLASS:
                print(f"[WARNING] Skipping class '{label}': Only found {num_samples} valid audio samples.")
            else:
                print(f"Class '{label}': Found {num_samples} valid audio samples. OK.")
                valid_class_labels.append((label, sample_files))
        
        print(f"\n--- Starting Data Extraction via FFmpeg ---")
        
        for label, sample_files in valid_class_labels:
            print(f"\nProcessing folder: {label}")
            for filename in sample_files:
                file_path = os.path.join(dataset_path, label, filename)
                data = self.extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(label)

        if not features: return None, None, None
        encoded_labels = self.encoder.fit_transform(labels)
        return np.array(features), to_categorical(encoded_labels), self.encoder.classes_

    def build_model(self, input_shape, num_classes):
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=input_shape), BatchNormalization(), Dropout(0.5),
            Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("\n--- Model Architecture ---")
        self.model.summary()

    def train(self, dataset_path):
        X, y, self.classes_ = self.load_data(dataset_path)
        if X is None:
            print("\nTraining aborted: No valid data was loaded.")
            return
        # --- Suppress a harmless tensorflow warning ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.build_model((X_train.shape[1],), y_train.shape[1])
            print("\n--- Starting Model Training ---")
            self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
            print("--- Model Training Complete ---\n")
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"Final Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    def save_model(self):
        if self.model:
            self.model.save(MODEL_FILE_NAME)
            np.save(CLASSES_FILE_NAME, self.encoder.classes_)
            print(f"Model saved to '{MODEL_FILE_NAME}' and '{CLASSES_FILE_NAME}'")

if __name__ == '__main__':
    DATASET_PATH = os.path.join('..', 'dataset')
    classifier = AudioClassifier()
    classifier.train(DATASET_PATH)
    classifier.save_model()