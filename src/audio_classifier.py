import os
import sys
import random
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

# ==============================================================================
# --- AI TRAINING CONTROL PANEL (FOR MANUAL RUNS) ---
# ==============================================================================
# These settings are used ONLY when you run 'python audio_classifier.py'
TRAINING_EPOCHS = 250
BATCH_SIZE = 32
MODEL_LAYERS = [512, 256, 128, 64]
DROPOUT_RATE = 0.5
LR_DROP_FACTOR = 0.5
LR_DROP_EPOCHS = 40
# ==============================================================================
# --- RANDOM SEARCH CONFIGURATION (FOR --random-search MODE) ---
# ==============================================================================
# The script will randomly pick values from these ranges when in search mode.
RANDOM_CONFIG = {
    'epochs': [150, 200, 250, 300, 350, 400],
    'batch_size': [32, 64],
    'layers': [
        [256, 128],
        [512, 256, 128],
        [512, 256, 128, 64],
        [768, 512, 256, 128],
        [768, 512, 256, 128, 64],
    ],
    'dropout': [0.4,0.45, 0.5,0.55, 0.6],
    'lr_drop_factor': [0.5,0.55, 0.6],
    'lr_drop_epochs': [30, 40, 50],
}
# ==============================================================================

MODEL_DIR = os.path.join('..', 'models')
DATA_DIR = os.path.join('..', 'data')
MODEL_FILE_NAME = os.path.join(MODEL_DIR, "audio_classifier_model.keras")
CLASSES_FILE_NAME = os.path.join(MODEL_DIR, "classes.npy")
PREPROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'preprocessed_data.npz')
BEST_ACCURACY_FILE = os.path.join(MODEL_DIR, "best_accuracy.txt")

def configure_hardware():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"INFO: Found {len(gpus)} compatible GPU(s). Training will use the GPU.")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("INFO: No compatible GPU found. Training will run on the CPU.")
    except Exception as e:
        print(f"WARNING: Could not configure GPU. Falling back to CPU. Error: {e}")

def get_random_config():
    """Generates a random set of training parameters."""
    config = {
        'epochs': random.choice(RANDOM_CONFIG['epochs']),
        'batch_size': random.choice(RANDOM_CONFIG['batch_size']),
        'layers': random.choice(RANDOM_CONFIG['layers']),
        'dropout': random.choice(RANDOM_CONFIG['dropout']),
        'lr_drop_factor': random.choice(RANDOM_CONFIG['lr_drop_factor']),
        'lr_drop_epochs': random.choice(RANDOM_CONFIG['lr_drop_epochs']),
    }
    return config

def train_with_config(config):
    """The main training function, now accepts a configuration dictionary."""
    # Unpack the config
    epochs = config['epochs']
    batch_size = config['batch_size']
    model_layers = config['layers']
    dropout_rate = config['dropout']
    lr_drop_factor = config['lr_drop_factor']
    lr_drop_epochs = config['lr_drop_epochs']

    print("\n" + "="*60)
    print(" " * 20 + "STARTING NEW TRAINING RUN")
    print("="*60)
    print("Current Configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    
    previous_best_accuracy = 0.0
    if os.path.exists(BEST_ACCURACY_FILE):
        with open(BEST_ACCURACY_FILE, 'r') as f:
            try: previous_best_accuracy = float(f.read())
            except: pass
    print(f"Previous best accuracy to beat: {previous_best_accuracy * 100:.2f}%")

    data = np.load(PREPROCESSED_DATA_FILE, allow_pickle=True)
    X, y_raw = data['X'], data['y']
    encoder = LabelEncoder()
    y = to_categorical(encoder.fit_transform(y_raw))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    input_shape = (X_train.shape[1],)
    num_classes = y_train.shape[1]
    
    model = Sequential()
    model.add(Dense(model_layers[0], activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    for neurons in model_layers[1:]:
        model.add(Dense(neurons, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def lr_scheduler(epoch, lr):
        if epoch > 0 and epoch % lr_drop_epochs == 0:
            return lr * lr_drop_factor
        return lr
    
    lr_callback = LearningRateScheduler(lr_scheduler)

    print(f"\n--- Starting Model Training ({epochs} Epochs) ---")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[lr_callback], verbose=2)
    
    print("--- Training Complete ---\n")
    loss, new_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Run Finished. Accuracy: {new_accuracy * 100:.2f}%")

    if new_accuracy > previous_best_accuracy:
        print(f"\n--- !!! NEW BEST MODEL FOUND !!! ---")
        print(f"Accuracy improved from {previous_best_accuracy * 100:.2f}% to {new_accuracy * 100:.2f}%. SAVING.")
        model.save(MODEL_FILE_NAME)
        np.save(CLASSES_FILE_NAME, encoder.classes_)
        with open(BEST_ACCURACY_FILE, 'w') as f:
            f.write(str(new_accuracy))
    else:
        print(f"--- Accuracy did not improve. Discarding model. ---")

if __name__ == '__main__':
    configure_hardware()
    
    if not os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"ERROR: Data file '{PREPROCESSED_DATA_FILE}' not found.")
        print("Please run 'python preprocess.py' first.")
        sys.exit(1)

    # --- Check for the special command-line flag ---
    if '--random-search' in sys.argv:
        print("\n--- ENTERING ENDLESS RANDOM SEARCH MODE ---")
        print("The script will now train models with random settings until you stop it with Ctrl+C.")
        while True:
            try:
                random_config = get_random_config()
                train_with_config(random_config)
            except KeyboardInterrupt:
                print("\n\nRandom search stopped by user. Exiting.")
                break
    else:
        # --- Run in normal, manual mode ---
        print("\n--- RUNNING IN MANUAL MODE ---")
        manual_config = {
            'epochs': TRAINING_EPOCHS,
            'batch_size': BATCH_SIZE,
            'layers': MODEL_LAYERS,
            'dropout': DROPOUT_RATE,
            'lr_drop_factor': LR_DROP_FACTOR,
            'lr_drop_epochs': LR_DROP_EPOCHS,
        }
        train_with_config(manual_config)