import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- THIS SCRIPT HAS NO LIBSROSA OR SUBPROCESS ---

# --- Configuration ---
MODEL_FILE_NAME = "audio_classifier_model.keras"
CLASSES_FILE_NAME = "classes.npy"
PREPROCESSED_DATA_FILE = 'preprocessed_data.npz'

def train_model():
    """Loads preprocessed data and trains the Keras model."""
    print("\n--- STAGE 2: TRAINING ---")
    
    # 1. Load the data created by preprocess.py
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"ERROR: Data file '{PREPROCESSED_DATA_FILE}' not found.")
        print("Please run 'python preprocess.py' first to generate the data.")
        return

    print(f"Loading data from '{PREPROCESSED_DATA_FILE}'...")
    data = np.load(PREPROCESSED_DATA_FILE, allow_pickle=True)
    X = data['X']
    y_raw = data['y']
    
    # 2. Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y = to_categorical(y_encoded)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Build and train the model
    input_shape = (X_train.shape[1],)
    num_classes = y_train.shape[1]
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape), BatchNormalization(), Dropout(0.5),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n--- Model Architecture ---")
    model.summary()

    print("\n--- Starting Model Training ---")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
    
    print("--- Model Training Complete ---\n")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    # 5. Save the final model and the class encoder
    model.save(MODEL_FILE_NAME)
    np.save(CLASSES_FILE_NAME, encoder.classes_)
    print(f"Model saved to '{MODEL_FILE_NAME}' and '{CLASSES_FILE_NAME}'")

if __name__ == '__main__':
    train_model()