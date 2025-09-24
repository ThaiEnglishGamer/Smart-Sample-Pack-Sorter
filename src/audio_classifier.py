import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler # NEW: For smarter training

# --- MODIFIED: Paths now point to the correct new folders ---
MODEL_DIR = os.path.join('..', 'models')
DATA_DIR = os.path.join('..', 'data')
MODEL_FILE_NAME = os.path.join(MODEL_DIR, "audio_classifier_model.keras")
CLASSES_FILE_NAME = os.path.join(MODEL_DIR, "classes.npy")
PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'preprocessed_data.npz')

# NEW: Learning Rate Scheduler function
def lr_scheduler(epoch, lr):
    """Gradually reduces the learning rate over epochs."""
    if epoch > 0 and epoch % 30 == 0:
        return lr * 0.5
    return lr

def train_model():
    """Loads preprocessed data and trains the Keras model."""
    print("\n--- STAGE 2: TRAINING (High Accuracy Mode) ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"ERROR: Data file '{PREPROCESSED_DATA_FILE}' not found.")
        print("Please run 'python preprocess.py' first to generate the data.")
        return

    print(f"Loading data from '{PREPROCESSED_DATA_FILE}'...")
    data = np.load(PREPROCESSED_DATA_FILE, allow_pickle=True)
    X = data['X']
    y_raw = data['y']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    y = to_categorical(y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    input_shape = (X_train.shape[1],)
    num_classes = y_train.shape[1]
    
    # --- MODIFIED: Deeper model architecture ---
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape), BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5), # New layer
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n--- Upgraded Model Architecture ---")
    model.summary()

    # NEW: Create the learning rate callback
    lr_callback = LearningRateScheduler(lr_scheduler)

    print("\n--- Starting Model Training (150 Epochs) ---")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')
        # --- MODIFIED: Increased epochs and added the callback ---
        model.fit(X_train, y_train, epochs=150, batch_size=32,
                  validation_data=(X_test, y_test),
                  callbacks=[lr_callback], verbose=2)
    
    print("--- Model Training Complete ---\n")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model Accuracy on Test Data: {accuracy * 100:.2f}%")

    model.save(MODEL_FILE_NAME)
    np.save(CLASSES_FILE_NAME, encoder.classes_)
    print(f"Model saved to '{MODEL_FILE_NAME}' and '{CLASSES_FILE_NAME}'")

if __name__ == '__main__':
    train_model()