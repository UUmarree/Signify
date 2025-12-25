import os
import shutil
import sys

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, DATA_PATH
except ImportError:
    print("❌ Config not found. Using default paths...")
    # Define your actions manually if config fails
    # ACTIONS = np.array(['fever', 'pain', ...]) 
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')

# --- CONFIGURATION ---
TARGET_MAX = 30  # The cap. Anything above this gets deleted.

def trim_dataset():
    print(f"✂️  Starting Dataset Trim (Cap: {TARGET_MAX} sequences)")
    print(f"📂 Target Folder: {DATA_PATH}")
    print("-" * 50)

    total_deleted = 0

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        
        if not os.path.exists(action_path):
            print(f"Skipping {action} (Folder not found)")
            continue

        # Get list of sequence folders (0, 1, 2...)
        # We must sort them numerically, or '10' comes before '2'
        try:
            sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f)) and f.isdigit()]
            sequences.sort(key=int) 
        except ValueError:
            print(f"⚠️  Warning: Non-numeric folders found in {action}. Skipping sort.")
            continue

        count = len(sequences)
        
        if count > TARGET_MAX:
            # Calculate how many to kill
            excess = count - TARGET_MAX
            print(f"🔴 {action.upper()}: Found {count} sequences. Deleting {excess}...")

            # We keep the first 50 (indices 0 to 49)
            # We delete everything from index 50 onwards
            folders_to_delete = sequences[TARGET_MAX:]
            
            for folder_name in folders_to_delete:
                folder_path = os.path.join(action_path, folder_name)
                try:
                    shutil.rmtree(folder_path)
                    total_deleted += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {folder_path}: {e}")
            
            print(f"   ✅ Trimmed down to {TARGET_MAX}.")
        else:
            print(f"🟢 {action.upper()}: count is {count} (Safe).")

    print("-" * 50)
    print(f"🎉 Cleanup Complete. Deleted {total_deleted} excess sequences.")
    print("👉 Now run 'python src/supplement_data.py' to fill the smaller classes up to 50.")

if __name__ == "__main__":
    trim_dataset()