import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
MODEL_FILE_NAME = "audio_classifier_model.keras"
CLASSES_FILE_NAME = "classes.npy"
AUDIO_TARGET_LENGTH = 2  # in seconds
MIN_SAMPLES_PER_CLASS = 10 # Minimum samples required to include a category in training

# ==============================================================================
#  CLASS 1: THE SORTER (Used by the GUI)
# ==============================================================================
class SampleSorter:
    def __init__(self, confidence_threshold=0.60):
        self.model = None
        self.classes = None
        self.confidence_threshold = confidence_threshold
        
        # --- Updated Keyword Map to match your folder structure ---
        self.keyword_map = {
            # Drums
            'kick': 'Drums_Kick', 'kik': 'Drums_Kick',
            'snare': 'Drums_Snare', 'snr': 'Drums_Snare',
            'clap': 'Drums_Clap', 'clp': 'Drums_Clap',
            'hat': 'Drums_Cymbals', 'hihat': 'Drums_Cymbals', 'cymbal': 'Drums_Cymbals', 'ride': 'Drums_Cymbals',
            'crash': 'Drums_Crash', 'crsh': 'Drums_Crash',
            'fill': 'Drums_Fills',
            'perc': 'Drums_Percussion',
            # Loops
            'drumloop': 'Loops_Drums', 'drum loop': 'Loops_Drums',
            'bassline': 'Loops_Synth_Bass', 'bass loop': 'Loops_Synth_Bass',
            'chord loop': 'Loops_Synth_Chords',
            'lead loop': 'Loops_Synth_Lead', 'melody': 'Loops_Synth_Lead',
            # Samples
            'piano': 'Sample_Piano',
            'synth': 'Sample_Synth',
            'bass': 'Sample_Synth_Bass',
            'shot': 'Sample_Oneshot',
            # FX
            'impact': 'FX_Impact', 'imp': 'FX_Impact',
            'riser': 'FX_Riser', 'rise': 'FX_Riser',
            'sweep': 'FX_Sweep', 'noise': 'FX_Sweep',
            'ambient': 'FX_Ambient', 'pad': 'FX_Ambient',
            'vocal': 'FX_Vocals', 'vox': 'FX_Vocals',
        }

    def load_model(self):
        """Loads the trained Keras model and class names from disk."""
        if os.path.exists(MODEL_FILE_NAME) and os.path.exists(CLASSES_FILE_NAME):
            self.model = tf.keras.models.load_model(MODEL_FILE_NAME)
            self.classes = np.load(CLASSES_FILE_NAME, allow_pickle=True)
            print("AI model and classes loaded successfully.")
            return True
        else:
            print("Model files not found. Please train the model first.")
            return False

    def predict_category(self, file_path):
        """
        Predicts category using the hybrid Filename + AI approach.
        Returns the category name (e.g., 'Drums_Kick') and the reason.
        """
        # --- Stage 1: Filename Analysis ---
        filename = os.path.basename(file_path).lower()
        for keyword, category in self.keyword_map.items():
            if keyword in filename:
                return category, "Keyword"

        # --- Stage 2: AI Audio Analysis (if no keyword found) ---
        if not self.model:
            return "_Uncategorized", "AI Model not loaded"

        features = self._extract_features(file_path)
        if features is None:
            return "_Uncategorized", "Feature Extraction Error"

        features = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features, verbose=0)[0]
        
        confidence = np.max(prediction)
        
        if confidence >= self.confidence_threshold:
            predicted_index = np.argmax(prediction)
            category_name = self.classes[predicted_index]
            return category_name, f"AI ({confidence*100:.0f}%)"
        else:
            return "_Uncategorized", f"AI Low Confidence ({confidence*100:.0f}%)"

    def _extract_features(self, file_path, sr=22050, n_mfcc=13):
        """Extracts numerical features from a single audio file for prediction."""
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            audio = librosa.util.fix_length(data=audio, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

# ==============================================================================
#  CLASS 2: THE TRAINER (Used for training the model from the command line)
# ==============================================================================
class AudioClassifier:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()

    def extract_features(self, file_path, sr=22050, n_mfcc=13):
        """This function is robust against file errors that can be caught."""
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            audio = librosa.util.fix_length(data=audio, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"SKIPPING corrupt or unreadable file: {os.path.basename(file_path)} | Error: {e}")
            return None

    def load_data(self, dataset_path):
        """
        Loads data, automatically ignoring corrupted files and classes with too few samples.
        """
        features, labels = [], []
        
        all_class_labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        valid_class_labels = []
        print("\n--- Analyzing Dataset ---")
        for label in all_class_labels:
            class_dir = os.path.join(dataset_path, label)
            num_samples = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
            if num_samples < MIN_SAMPLES_PER_CLASS:
                print(f"[WARNING] Skipping class '{label}': Only found {num_samples} samples (minimum is {MIN_SAMPLES_PER_CLASS}).")
            else:
                print(f"Class '{label}': Found {num_samples} samples. OK.")
                valid_class_labels.append(label)
        
        print(f"\nTraining with {len(valid_class_labels)} valid classes.")
        
        for label in valid_class_labels:
            class_dir = os.path.join(dataset_path, label)
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path):
                    # --- DEBUGGING PRINT STATEMENT ---
                    # This will force the program to print the file path just before trying to process it.
                    # The last path printed before a crash is the problematic file.
                    print(f"--> Attempting to process: {file_path}", flush=True)
                    
                    data = self.extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(label)

        if not features:
            return None, None, None

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
            print("\nTraining aborted: No valid data was loaded. Please check your dataset folder and sample counts.")
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
    DATASET_PATH = os.path.join('..', 'dataset')
    
    classifier = AudioClassifier()
    classifier.train(DATASET_PATH)
    classifier.save_model()