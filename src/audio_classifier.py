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
AUDIO_TARGET_LENGTH = 2 # in seconds

# ==============================================================================
#  CLASS 1: THE SORTER (Used by the GUI)
# ==============================================================================
class SampleSorter:
    def __init__(self, confidence_threshold=0.60):
        self.model = None
        self.classes = None
        self.confidence_threshold = confidence_threshold
        
        # --- The heart of the filename analysis ---
        # Maps keywords to your final folder names. Order matters: more specific keywords first.
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
            'drumloop': 'Loop_Drums', 'drum loop': 'Loop_Drums',
            'bassline': 'Loops_Synth_Bass', 'bass loop': 'Loops_Synth_Bass',
            'chord loop': 'Loops_Synth_Chords',
            'lead loop': 'Loops_Synth_Lead', 'melody': 'Loops_Synth_Lead',
            # Samples
            'piano': 'Sample_Piano',
            'synth': 'Sample_Synth',
            'bass': 'Sample_Synth_Bass', # Must be after bassline/bass loop to not misclassify loops
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
        Returns the category name (e.g., 'Drums_Kick')
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

        features = np.expand_dims(features, axis=0) # Reshape for prediction
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
    # This class is largely the same as before, used only for training.
    # Its purpose is to create the .keras and .npy files that SampleSorter uses.
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()

    def extract_features(self, file_path, sr=22050, n_mfcc=13):
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            audio = librosa.util.fix_length(data=audio, size=target_samples)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            return np.mean(mfccs.T, axis=0)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def load_data(self, dataset_path):
        features, labels = [], []
        class_labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found classes: {class_labels}")
        for label in class_labels:
            class_dir = os.path.join(dataset_path, label)
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path):
                    data = self.extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(label)
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
        print(self.model.summary())

    def train(self, dataset_path):
        X, y, self.classes_ = self.load_data(dataset_path)
        if len(X) == 0:
            print("No data loaded. Please check your dataset folder.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.build_model((X_train.shape[1],), y_train.shape[1])
        print("\n--- Starting Model Training ---")
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        print("--- Model Training Complete ---\n")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

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