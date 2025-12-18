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

# Input and Output Paths
INPUT_MODEL = os.path.join(ROOT_DIR, 'models', 'action.h5')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'web_app', 'public', 'model')

def convert():
    print("🚀 Starting Model Conversion (Python API Mode)...")
    print(f"   Input:  {INPUT_MODEL}")
    print(f"   Output: {OUTPUT_DIR}")

    # 1. Check if input exists
    if not os.path.exists(INPUT_MODEL):
        print(f"❌ Error: Input model not found at {INPUT_MODEL}")
        print("   Did you run train_model.py?")
        return

    # 2. Clear/Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
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