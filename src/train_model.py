from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
import numpy as np
import os
import sys
import tensorflow as tf

# --- DIAGNOSTIC IMPORT BLOCK ---
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecay
    from tensorflow.keras.regularizers import l2 # Added for extra stability
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
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path): continue
        
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

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# --- 2. SPLIT DATA (15% Test, 10% Val, 75% Train) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.10, random_state=42)

print(f"📊 Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# --- 2.5 CLASS WEIGHTS (Inversely Proportional) ---
y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_integers), 
    y=y_integers
)
class_weight_dict = dict(enumerate(class_weights))
print(f"⚖️  Class Weights: {class_weight_dict}")

# --- 3. BUILD AGGRESSIVELY REGULARIZED MODEL ---
# Reverted to Standard LSTM because Bidirectional was overfitting (95% train vs 48% val)
print("🧠 Building Neural Network (High Regularization Architecture)...")
model = Sequential()

# Layer 1: LSTM (32 units) + L2 Reg
# We added kernel_regularizer=l2(0.001) to punish large weights
model.add(LSTM(32, return_sequences=True, activation='tanh', 
               input_shape=(SEQUENCE_LENGTH, 1662), kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5)) # Increased to 50% to force generalization

# Layer 2: LSTM (32 units)
model.add(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Layer 3: Dense
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(ACTIONS.shape[0], activation='softmax'))

# --- 3.5 COSINE ANNEALING SCHEDULER ---
# Parameters from prompt: Initial LR = 1e-3, T_max = 150 (mapped to decay steps)
EPOCHS = 200
BATCH_SIZE = 32
# Calculate total steps for decay
# We set decay_steps to cover the full 200 epochs to ensure smooth decline
steps_per_epoch = len(X_train) // BATCH_SIZE
decay_steps = steps_per_epoch * EPOCHS 

lr_schedule = CosineDecay(
    initial_learning_rate=0.001, # 1e-3
    decay_steps=decay_steps,
    alpha=0.0 # Final LR goes to 0
)

opt = Adam(learning_rate=lr_schedule, clipnorm=1.0)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. TRAIN ---
log_dir = os.path.join(ROOT_DIR, 'logs')
os.makedirs(MODELS_PATH, exist_ok=True)

tb_callback = TensorBoard(log_dir=log_dir)
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(MODELS_PATH, 'best_model.h5'),
    monitor='val_categorical_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("🏋️ Starting Training with Cosine Annealing...")
# Removed EarlyStopping so the Cosine Decay can run its full course
model.fit(X_train, y_train, epochs=EPOCHS, 
          batch_size=BATCH_SIZE,
          callbacks=[tb_callback, checkpoint_callback], 
          validation_data=(X_val, y_val),
          class_weight=class_weight_dict)

# --- 5. SAVE ---
model.save(os.path.join(MODELS_PATH, 'action.h5'))