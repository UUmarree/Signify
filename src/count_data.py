import os
import sys
import numpy as np

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, DATA_PATH
except ImportError:
    print("❌ Config not found. Attempting fallback...")
    ACTIONS = np.array([ 'pain', 'medicine', 'hello', 'thankyou', 
       'yes', 'no', 'help', 'nothing'])
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')

print(f"📊 Counting sequences in: {DATA_PATH}\n")
print(f"{'ACTION':<15} | {'COUNT':<10} | {'STATUS'}")
print("-" * 45)

total_sequences = 0
missing_actions = []
low_data_actions = []

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    
    if not os.path.exists(action_path):
        print(f"{action:<15} | {'0':<10} | ❌ Missing Folder")
        missing_actions.append(action)
        continue
        
    # Get all sequence folders (0, 1, 2...)
    # We only count actual folders, not random files
    sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    count = len(sequences)
    total_sequences += count
    
    status = ""
    if count == 0:
        status = "❌ Empty"
        missing_actions.append(action)
    elif count < 30:
        status = f"⚠️ Low ({30-count} needed)"
        low_data_actions.append(action)
    else:
        status = "✅ Healthy"
        
    print(f"{action:<15} | {count:<10} | {status}")

print("-" * 45)
print(f"TOTAL DATASET SIZE: {total_sequences} videos")

# --- ACTION PLAN ---
print("\n📋 DIAGNOSIS:")
if not missing_actions and not low_data_actions:
    print("🎉 Dataset is perfect! You are ready to train.")
else:
    if missing_actions:
        print(f"🔴 CRITICAL: You have {len(missing_actions)} missing words.")
        print(f"   Target: {missing_actions}")
        print("   Action: Run 'python src/supplement_data.py' immediately.")
    
    if low_data_actions:
        print(f"🟡 WARNING: You have {len(low_data_actions)} words with low data.")
        print(f"   Target: {low_data_actions}")
        print("   Action: Run 'python src/supplement_data.py' to top them up.")