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
MODEL_FILE_NAME = "audio_classifier_model.keras" # Using the modern .keras format
AUDIO_TARGET_LENGTH = 2 # in seconds, ensures all audio clips are the same size for the AI

class AudioClassifier:
    def __init__(self):
        self.model = None
        self.encoder = LabelEncoder()

    def extract_features(self, file_path, sr=22050, n_mfcc=13):
        """Extracts numerical features from an audio file."""
        try:
            audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
            
            # Pad or truncate the audio to a fixed length
            target_samples = AUDIO_TARGET_LENGTH * sample_rate
            if len(audio) > target_samples:
                audio = audio[:target_samples]
            else:
                audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')

            # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def load_data(self, dataset_path):
        """Loads data from the dataset folder structure."""
        features = []
        labels = []
        
        # The subdirectories (e.g., 'Kick', 'Snare') are our class names
        class_labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found classes: {class_labels}")

        for label in class_labels:
            class_dir = os.path.join(dataset_path, label)
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                
                # Check if it's a file before processing
                if os.path.isfile(file_path):
                    data = self.extract_features(file_path)
                    if data is not None:
                        features.append(data)
                        labels.append(label)

        # Convert labels to numbers (e.g., Kick=0, Snare=1)
        encoded_labels = self.encoder.fit_transform(labels)
        categorical_labels = to_categorical(encoded_labels)
        
        return np.array(features), categorical_labels, self.encoder.classes_

    def build_model(self, input_shape, num_classes):
        """Builds and compiles the neural network model."""
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax') # Softmax for multi-class classification
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, dataset_path):
        """Loads data, builds the model, and trains it."""
        X, y, self.classes_ = self.load_data(dataset_path)
        
        if len(X) == 0:
            print("No data loaded. Please check your dataset folder.")
            return

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        input_shape = (X_train.shape[1],)
        num_classes = y_train.shape[1]

        self.build_model(input_shape, num_classes)
        
        print("\n--- Starting Model Training ---")
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        print("--- Model Training Complete ---\n")

        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    def save_model(self):
        """Saves the trained model and the label encoder."""
        if self.model:
            self.model.save(MODEL_FILE_NAME)
            np.save('classes.npy', self.encoder.classes_)
            print(f"Model saved to {MODEL_FILE_NAME}")

# --- This part allows us to run this file directly to train the model ---
if __name__ == '__main__':
    # The path to your dataset folder. The '..' moves up one directory from 'src'.
    DATASET_PATH = os.path.join('..', 'dataset')

    classifier = AudioClassifier()
    # This single line will load data, build the model, and train it.
    classifier.train(DATASET_PATH)
    # After training, it saves the result.
    classifier.save_model()