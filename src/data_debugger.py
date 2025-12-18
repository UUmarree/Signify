import numpy as np
import os
import sys

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH
except ImportError:
    # Fallback
    ACTIONS = np.array(['fever', 'pain', 'medicine'])
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
    SEQUENCE_LENGTH = 30

print(f"🕵️ Deep Inspecting data in: {DATA_PATH}\n")

total_issues = 0

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    
    if not os.path.exists(action_path):
        print(f"❌ Missing folder for: {action}")
        continue

    sequences = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]
    
    print(f"📂 Analyzing '{action.upper()}' ({len(sequences)} sequences found)...")
    
    good_count = 0
    bad_count = 0
    
    for seq in sequences:
        seq_path = os.path.join(action_path, seq)
        frames = []
        is_bad = False
        
        # 1. Load all frames
        for i in range(SEQUENCE_LENGTH):
            frame_file = os.path.join(seq_path, f"{i}.npy")
            if not os.path.exists(frame_file):
                print(f"   ⚠️ Seq {seq}: Missing frame {i}")
                is_bad = True
                break
            frames.append(np.load(frame_file))
            
        if is_bad:
            bad_count += 1
            total_issues += 1
            continue
            
        # Convert to one big array for this video (30, 1662)
        video_data = np.array(frames)
        
        # 2. Check for Zeros (MediaPipe missed detections)
        if np.max(video_data) == 0:
            print(f"   ⚠️ Seq {seq}: ALL ZEROS (No body detected)")
            is_bad = True
        
        # 3. Check for Static Data
        elif np.var(video_data) == 0:
            print(f"   ⚠️ Seq {seq}: STATIC (Frames are identical copies)")
            is_bad = True
            
        # 4. Check for NaNs
        elif np.isnan(video_data).any():
            print(f"   ⚠️ Seq {seq}: Contains NaNs (Corrupt Data)")
            is_bad = True

        # 5. NEW: Check for MISSING HANDS specifically
        # Data Layout: Pose(132) + Face(1404) + LH(63) + RH(63)
        # Left Hand starts at index 1536. Right Hand starts at 1599.
        else:
            left_hand = video_data[:, 1536:1599]
            right_hand = video_data[:, 1599:]

            # If max value is 0, it means the hand was never detected in the whole video
            lh_missing = np.max(left_hand) == 0
            rh_missing = np.max(right_hand) == 0

            if lh_missing and rh_missing:
                print(f"   ⚠️ Seq {seq}: NO HANDS (Face present, but hands missing)")
                is_bad = True

        if is_bad:
            bad_count += 1
            total_issues += 1
        else:
            good_count += 1
            
    print(f"   ✅ Good: {good_count} | ❌ Bad: {bad_count}")

print(f"\n--- Inspection Complete ---")
if total_issues > 0:
    print(f"⚠️ Found {total_issues} problematic sequences. You should delete these folders before training.")
else:
    print("🎉 All data looks healthy!")