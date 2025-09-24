import os
import time
import numpy as np
import librosa
# --- NEW: Import the soundfile library to check for a specific error ---
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- Configuration (remains the same) ---
MODEL_FILE_NAME = "audio_classifier_model.keras"
CLASSES_FILE_NAME = "classes.npy"
AUDIO_TARGET_LENGTH = 2
MIN_SAMPLES_PER_CLASS = 10
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg')


# ==============================================================================
#  CLASS 1: THE SORTER
# ==============================================================================
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
        # --- MODIFIED: Use the new robust feature extraction ---
        features = self._robust_extract_features(file_path)
        if features is None: return "_Uncategorized", "Feature Extraction Error"
        features = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features, verbose=0)[0]
        confidence = np.max(prediction)
        if confidence >= self.confidence_threshold:
            return self.classes[np.argmax(prediction)], f"AI ({confidence*100:.0f}%)"
        else:
            return "_Uncategorized", f"AI Low Confidence ({confidence*100:.0f}%)"

    # --- NEW: Renamed to _robust_extract_features and updated logic ---
    def _robust_extract_features(self, file_path, sr=22050, n_mfcc=13):
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            audio = librosa.util.fix_length(data=audio, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"      -> SKIPPING (Readable but corrupt): {os.path.basename(file_path)} | Error: {e}")
            return None


# ==============================================================================
#  CLASS 2: THE TRAINER
# ==============================================================================
class AudioClassifier:
    def __init__(self):
        self.model, self.encoder = None, LabelEncoder()

    # --- MODIFIED: This is the key change to make our loading more robust ---
    def extract_features(self, file_path, sr=22050, n_mfcc=13):
        """
        Attempts to load an audio file with librosa's default engine,
        and falls back to a more robust engine if it fails.
        """
        try:
            # First, try the default (often faster) method
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
        except Exception:
            # If default fails, it might be a format issue.
            # audioread with ffmpeg is more robust.
            print(f"      -> Default load failed. Retrying with audioread/ffmpeg for: {os.path.basename(file_path)}")
            try:
                # Use audioread engine, which should now be backed by FFmpeg
                audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast', engine='audioread')
            except Exception as e:
                print(f"      -> SKIPPING: Could not read file even with fallback. | Error: {e}")
                return None
        
        # Once loaded (by either method), process as before
        try:
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            audio = librosa.util.fix_length(data=audio, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"      -> SKIPPING: Feature extraction failed after load. | Error: {e}")
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
        
        print(f"\n--- Starting Data Extraction ---")
        
        for label, sample_files in valid_class_labels:
            print(f"\nProcessing folder: {label}")
            for filename in sample_files:
                file_path = os.path.join(dataset_path, label, filename)
                # The debugging print is still useful to see what's happening
                print(f"  -> Reading: {filename}", flush=True)
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
    # Need to make sure audioread is installed
    try:
        import audioread
    except ImportError:
        print("\n\nERROR: The 'audioread' package is not installed.")
        print("Please install it to enable robust audio file loading:")
        print("pip install audioread\n\n")
        exit()

    DATASET_PATH = os.path.join('..', 'dataset')
    classifier = AudioClassifier()
    classifier.train(DATASET_PATH)
    classifier.save_model()