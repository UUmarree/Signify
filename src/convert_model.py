import os
import sys
import shutil
from unittest.mock import MagicMock

# --- WINDOWS FIX ---
sys.modules['tensorflow_decision_forests'] = MagicMock()

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
except ImportError as e:
    print(f"❌ Library Import Error: {e}")
    sys.exit(1)

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# --- CONFIGURATION ---
# 🟢 MANUAL OVERRIDE: Set this to the filename you want to deploy.
# Example: 'golden_model_v1.h5' (To revert to v1)
# Example: None (To automatically use the latest best_model.h5)
FORCE_MODEL_FILENAME = 'golden_model_v1.h5' 

# Standard paths
BEST_MODEL = os.path.join(MODELS_DIR, 'best_model.h5')
FINAL_MODEL = os.path.join(MODELS_DIR, 'action.h5')

# --- SELECTION LOGIC ---
INPUT_MODEL = None

if FORCE_MODEL_FILENAME:
    forced_path = os.path.join(MODELS_DIR, FORCE_MODEL_FILENAME)
    if os.path.exists(forced_path):
        INPUT_MODEL = forced_path
        print(f"🔒 Forced Selection: {FORCE_MODEL_FILENAME}")
    else:
        print(f"❌ Error: Forced model '{FORCE_MODEL_FILENAME}' not found in models/ folder.")
        print("   Available models:")
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.h5'): print(f"   - {f}")
        sys.exit(1)
elif os.path.exists(BEST_MODEL):
    INPUT_MODEL = BEST_MODEL
    print("⭐ Selected: best_model.h5 (Peak Training Performance)")
elif os.path.exists(FINAL_MODEL):
    INPUT_MODEL = FINAL_MODEL
    print("⚠️ Selected: action.h5 (Final Epoch - might be overfitted)")
else:
    print("❌ Error: No models found in 'models/' folder.")
    sys.exit(1)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'web_app', 'public', 'model')

def get_next_version_name(models_dir, base_name="golden_model_v"):
    v = 1
    while True:
        name = f"{base_name}{v}.h5"
        full_path = os.path.join(models_dir, name)
        if not os.path.exists(full_path):
            return name, full_path
        v += 1

def convert():
    print("🚀 Starting Model Conversion...")
    print(f"   Input:  {INPUT_MODEL}")
    print(f"   Output: {OUTPUT_DIR}")

    # 1. Auto-Backup (Only if using a fresh training run)
    if not FORCE_MODEL_FILENAME and "best_model.h5" in INPUT_MODEL:
        backup_name, backup_path = get_next_version_name(MODELS_DIR)
        print(f"💾 Saving safety backup: {backup_name}")
        shutil.copy2(INPUT_MODEL, backup_path)

    # 2. Clear/Create output directory
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"⚠️ Warning: Could not delete old folder: {e}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # 3. Load & Convert
        print("   Loading Keras model...")
        model = tf.keras.models.load_model(INPUT_MODEL)

        print("   Converting to TFJS format...")
        tfjs.converters.save_keras_model(model, OUTPUT_DIR)
        
        print("\n✅ Conversion Successful!")
        print(f"   Files created in {OUTPUT_DIR}:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"   - {f}")
        print("\n👉 Next Step: Restart 'npm start' to clear the cache!")

    except Exception as e:
        print(f"\n❌ Conversion Failed: {e}")

if __name__ == "__main__":
    convert()