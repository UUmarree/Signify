import os
import sys
import shutil
from unittest.mock import MagicMock

# --- WINDOWS FIX ---
# This prevents the "inference.so not found" crash by mocking the broken library
# before tensorflowjs tries to load it.
sys.modules['tensorflow_decision_forests'] = MagicMock()

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
except ImportError as e:
    print(f"❌ Library Import Error: {e}")
    print("   Run: pip install tensorflow tensorflowjs")
    sys.exit(1)

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# --- CONFIGURATION ---
VERIFIED_MODEL = os.path.join(ROOT_DIR, 'models', 'model_11words_80acc.h5')
BEST_MODEL = os.path.join(ROOT_DIR, 'models', 'best_model.h5')
FINAL_MODEL = os.path.join(ROOT_DIR, 'models', 'action.h5')

# Logic to pick the best available model
# ⚠️ UPDATE: We commented out the VERIFIED_MODEL check to ensure we use the
# FRESHLY trained model (best_model.h5).
# if os.path.exists(VERIFIED_MODEL):
#    INPUT_MODEL = VERIFIED_MODEL
#    print("🏆 Selected: model_11words_80acc.h5 (Verified High Accuracy)")
if os.path.exists(BEST_MODEL):
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
    """Finds the next available version number (v1, v2, v3...)"""
    v = 1
    while True:
        name = f"{base_name}{v}.h5"
        full_path = os.path.join(models_dir, name)
        if not os.path.exists(full_path):
            return name, full_path
        v += 1

def convert():
    print("🚀 Starting Model Conversion (Python API Mode)...")
    print(f"   Input:  {INPUT_MODEL}")
    print(f"   Output: {OUTPUT_DIR}")

    # 1. Check if input exists
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Error: Input model not found at {INPUT_MODEL}")
        return

    # 1.1 AUTO-BACKUP (Versioning System)
    # We automatically create a 'Golden' copy with a version number.
    if "best_model.h5" in INPUT_MODEL:
        models_dir = os.path.join(ROOT_DIR, 'models')
        backup_name, backup_path = get_next_version_name(models_dir)
        
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
        # 3. Load the Keras Model
        print("   Loading Keras model...")
        model = tf.keras.models.load_model(INPUT_MODEL)

        # 4. Save using TensorFlow.js
        print("   Converting to TFJS format...")
        tfjs.converters.save_keras_model(model, OUTPUT_DIR)
        
        print("\n✅ Conversion Successful!")
        print(f"   Files created in {OUTPUT_DIR}:")
        for f in os.listdir(OUTPUT_DIR):
            print(f"   - {f}")
        print("\n👉 Next Step: Your React App can now load this model!")

    except Exception as e:
        print(f"\n❌ Conversion Failed: {e}")
        # Fallback advice
        if "tensorflow_decision_forests" in str(e):
            print("\n💡 TIP: Try running this command in your terminal to remove the broken package:")
            print("   pip uninstall tensorflow-decision-forests -y")

if __name__ == "__main__":
    convert()