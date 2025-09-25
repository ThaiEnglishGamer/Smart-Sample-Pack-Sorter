import os
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# --- Configuration ---
MODEL_DIR = os.path.join('..', 'models')
DATA_DIR = os.path.join('..', 'data')
MODEL_FILE_NAME = os.path.join(MODEL_DIR, "audio_classifier_model.keras")
CLASSES_FILE_NAME = os.path.join(MODEL_DIR, "classes.npy")
PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'preprocessed_data.npz')
# --- NEW: File to store the best model's accuracy ---
BEST_ACCURACY_FILE = os.path.join(MODEL_DIR, "best_accuracy.txt")

def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 30 == 0:
        return lr * 0.5
    return lr

def train_model():
    """Loads preprocessed data and trains the Keras model,
    only saving if accuracy improves."""
    print("\n--- STAGE 2: TRAINING (High Accuracy Mode) ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"ERROR: Data file '{PREPROCESSED_DATA_FILE}' not found.")
        print("Please run 'python preprocess.py' first to generate the data.")
        return

    # --- NEW: Load the previous best accuracy ---
    previous_best_accuracy = 0.0
    if os.path.exists(BEST_ACCURACY_FILE):
        with open(BEST_ACCURACY_FILE, 'r') as f:
            try:
                previous_best_accuracy = float(f.read())
            except ValueError:
                previous_best_accuracy = 0.0
    print(f"Previous best model accuracy: {previous_best_accuracy * 100:.2f}%")

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
    
    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape), BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n--- Upgraded Model Architecture ---")
    model.summary()

    lr_callback = LearningRateScheduler(lr_scheduler)

    print("\n--- Starting Model Training (150 Epochs) ---")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')
        model.fit(X_train, y_train, epochs=150, batch_size=32,
                  validation_data=(X_test, y_test),
                  callbacks=[lr_callback], verbose=2)
    
    print("--- Model Training Complete ---\n")
    loss, new_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Current Model Accuracy on Test Data: {new_accuracy * 100:.2f}%")

    # --- NEW: Compare and save only if better ---
    if new_accuracy > previous_best_accuracy:
        print(f"\n--- NEW BEST MODEL FOUND! ---")
        print(f"Accuracy improved from {previous_best_accuracy * 100:.2f}% to {new_accuracy * 100:.2f}%. Saving model.")
        
        # Save the new best model, classes, and accuracy score
        model.save(MODEL_FILE_NAME)
        np.save(CLASSES_FILE_NAME, encoder.classes_)
        with open(BEST_ACCURACY_FILE, 'w') as f:
            f.write(str(new_accuracy))
    else:
        print(f"\n--- Training complete, but new model accuracy did not improve. ---")
        print(f"Current accuracy ({new_accuracy * 100:.2f}%) did not beat the previous best ({previous_best_accuracy * 100:.2f}%).")
        print("The old model has been kept.")

if __name__ == '__main__':
    train_model()