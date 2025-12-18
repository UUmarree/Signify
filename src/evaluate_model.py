from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import sys

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, SEQUENCE_LENGTH
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
    MODELS_PATH = os.path.join(ROOT_DIR, 'models')
except ImportError:
    print("❌ Config not found.")
    sys.exit(1)

# --- 1. LOAD DATA (Same logic as Training) ---
print(f"🚀 Loading data for evaluation...")
label_map = {label:num for num, label in enumerate(ACTIONS)}
sequences, labels = [], []

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path): continue
    
    sequence_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    for sequence in sequence_folders:
        window = []
        is_valid = True
        for frame_num in range(SEQUENCE_LENGTH):
            res_path = os.path.join(action_path, sequence, "{}.npy".format(frame_num))
            if not os.path.exists(res_path):
                is_valid = False
                break
            window.append(np.load(res_path))
        
        if is_valid:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# --- 2. RECREATE SPLIT (Critical for fair testing) ---
# We use random_state=42 to get the EXACT same 10% test set as training
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"📊 Evaluation Set Size: {len(X_test)} videos")

# --- 3. LOAD MODEL & PREDICT ---
model_path = os.path.join(MODELS_PATH, 'action.h5')
if not os.path.exists(model_path):
    print("❌ Model not found. Train it first!")
    sys.exit(1)

print("🧠 Loading model...")
model = load_model(model_path)

print("🔮 Predicting...")
yhat = model.predict(X_test)

# Convert from One-Hot (e.g., [0, 1, 0]) to Labels (e.g., 1)
ytrue = np.argmax(y_test, axis=1)
ypred = np.argmax(yhat, axis=1)

# --- 4. PRINT METRICS ---

print("\n" + "="*60)
print("FINAL EVALUATION REPORT")
print("="*60)

# Accuracy
acc = accuracy_score(ytrue, ypred)
print(f"🏆 Overall Accuracy: {acc*100:.2f}%")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
print("(Rows = Actual, Columns = Predicted)")
cm = confusion_matrix(ytrue, ypred)
print(cm)

# Detailed Report (Precision, Recall, F1)
print("\n--- Classification Report ---")
print(classification_report(ytrue, ypred, target_names=ACTIONS))
print("="*60)
print("\nInterpretation Guide:")
print("- Precision: When AI predicted 'Fever', how often was it actually Fever?")
print("- Recall:    Out of all actual 'Fever' videos, how many did AI catch?")
print("- F1-Score:  The balance between Precision and Recall (Best metric for FYP).")