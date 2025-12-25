import os
import shutil
import numpy as np
import sys

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, SEQUENCE_LENGTH, DATA_PATH
except ImportError:
    # Fallback
    ACTIONS = np.array([ 'pain', 'medicine', 'hello', 'thankyou', 
       'yes', 'no', 'help', 'nothing'])
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
    SEQUENCE_LENGTH = 30

print(f"🧹 Starting Cleanup in: {DATA_PATH}")
deleted_count = 0

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
        
    sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    for seq in sequences:
        seq_path = os.path.join(action_path, seq)
        should_delete = False
        
        # Check 1: Are there 30 frames?
        files = os.listdir(seq_path)
        if len(files) < SEQUENCE_LENGTH:
            print(f"   🗑️ Deleting {action}/{seq}: Only has {len(files)} frames (Need {SEQUENCE_LENGTH})")
            should_delete = True
            
        # Check 2: Are frames empty? (Optional deep check)
        if not should_delete:
            try:
                # Check just the first frame to save time
                first_frame = np.load(os.path.join(seq_path, "0.npy"))
                if first_frame.size == 0:
                    print(f"   🗑️ Deleting {action}/{seq}: Data is empty")
                    should_delete = True
            except Exception:
                 print(f"   🗑️ Deleting {action}/{seq}: Read error")
                 should_delete = True

        if should_delete:
            try:
                shutil.rmtree(seq_path)
                deleted_count += 1
            except Exception as e:
                print(f"      ❌ Error deleting: {e}")

print(f"\n✨ Cleanup Complete. Deleted {deleted_count} corrupt sequences.")