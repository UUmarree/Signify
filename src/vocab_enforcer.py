import os
import shutil
import sys
import numpy as np

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, DATA_PATH, NO_SEQUENCES
except ImportError:
    print("❌ Config not found.")
    sys.exit(1)

print(f"👮 Starting Vocabulary Enforcement in: {DATA_PATH}")
print(f"🎯 Target Vocabulary ({len(ACTIONS)} words): {ACTIONS}")
print(f"🎯 Target Quantity: {NO_SEQUENCES} videos per word")
print("-" * 50)

# 1. DELETE GHOST FOLDERS (Clean the house)
if os.path.exists(DATA_PATH):
    existing_folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]

    for folder in existing_folders:
        if folder not in ACTIONS:
            folder_path = os.path.join(DATA_PATH, folder)
            print(f"🗑️  Deleting GHOST folder: '{folder}' (Not in config)")
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"   ❌ Error deleting {folder}: {e}")
else:
    print(f"❌ Error: Data path {DATA_PATH} does not exist.")
    sys.exit(1)

# 2. CHECK TARGET FOLDERS (Find the gaps)
print("-" * 50)
missing_count = 0
low_count = 0

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    
    if not os.path.exists(action_path):
        print(f"❌ MISSING: '{action}' (Folder does not exist)")
        missing_count += 1
        continue
        
    sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    count = len(sequences)
    
    if count == 0:
        print(f"❌ EMPTY:   '{action}' has 0 sequences.")
        missing_count += 1
    elif count < NO_SEQUENCES:
        print(f"⚠️  LOW:     '{action}' has {count} sequences (Need {NO_SEQUENCES}).")
        low_count += 1
    else:
        print(f"✅ OK:      '{action}' has {count} sequences.")

print("-" * 50)
if missing_count > 0 or low_count > 0:
    print("🛑 STOP. Do not train.")
    print(f"   You have {missing_count} missing classes and {low_count} low-data classes.")
    print("   👉 Run 'python src/supplement_data.py' immediately to fix this.")
else:
    print("🚀 All systems go. You can Train now.")