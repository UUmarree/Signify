from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys

# --- DIAGNOSTIC IMPORT BLOCK ---
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    print("✅ TensorFlow imported successfully.")
except ImportError as e:
    print("\n❌ CRITICAL ERROR: TensorFlow is not installed or cannot be found.")
    sys.exit(1)

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
    MODELS_PATH = os.path.join(ROOT_DIR, 'models')
except ImportError:
    print("❌ Config not found.")
    sys.exit(1)

# --- 1. LOAD DATA ---
print(f"🚀 Loading data from: {DATA_PATH}")

label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    print(f"   📂 Loading '{action}'...")
    action_path = os.path.join(DATA_PATH, action)
    
    if not os.path.exists(action_path):
        continue
        
    sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    for sequence in sequence_folders:
        window = []
        is_valid_sequence = True
        for frame_num in range(SEQUENCE_LENGTH):
            res_path = os.path.join(action_path, sequence, "{}.npy".format(frame_num))
            if not os.path.exists(res_path):
                is_valid_sequence = False
                break
            res = np.load(res_path)
            window.append(res)
            
        if is_valid_sequence:
            sequences.append(window)
            labels.append(label_map[action])

print(f"📊 Total Dataset Size: {len(sequences)} videos.")

if len(sequences) == 0:
    print("❌ CRITICAL ERROR: No data loaded.")
    sys.exit(1)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# --- 2. DATA SPLITTING (10% Test, 5% Val, 85% Train) ---
print("✂️ Splitting Data...")

# Step A: Split off 10% for Testing (Leaves 90% for Train+Val)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Step B: Split off 5% of the TOTAL (which is ~5.55% of the remaining 90%) for Validation
# Calculation: 0.05 / 0.90 = 0.0555
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.0555, random_state=42)

print(f"   Training Set:   {len(X_train)} sequences (85%)")
print(f"   Validation Set: {len(X_val)} sequences (5%)")
print(f"   Testing Set:    {len(X_test)} sequences (10%)")

# --- 3. BUILD MODEL ---
print("🧠 Building Neural Network...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 1662)))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. TRAIN ---
log_dir = os.path.join(ROOT_DIR, 'logs')
os.makedirs(MODELS_PATH, exist_ok=True)

tb_callback = TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODELS_PATH, 'best_model.h5'),
    monitor='val_categorical_accuracy', # Monitor Validation accuracy now!
    save_best_only=True,
    mode='max',
    verbose=1
)

print("🏋️ Starting Training...")
# Note: validation_data is passed explicitly here
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, checkpoint_callback], validation_data=(X_val, y_val))

# --- 5. FINAL TEST ---
print("\n--- Final Test Evaluation (on the 10% Hold-out Set) ---")
final_model_path = os.path.join(MODELS_PATH, 'action.h5')
model.save(final_model_path)

if len(X_test) > 0:
    res = model.predict(X_test)
    
    # Calculate overall accuracy
    correct = 0
    for i in range(len(y_test)):
        if np.argmax(res[i]) == np.argmax(y_test[i]):
            correct += 1
    
    acc = correct / len(y_test)
    print(f"🏆 Final Test Accuracy: {acc*100:.2f}%")