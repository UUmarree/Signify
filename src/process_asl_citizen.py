import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import sys
import glob

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    # IMPORT NO_SEQUENCES TO PREVENT OVER-COLLECTION
    from src.config import ACTIONS, SEQUENCE_LENGTH, NO_SEQUENCES
    OUTPUT_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
except ImportError:
    print("❌ Config not found. Please run from project root.")
    sys.exit(1)

# --- CONFIGURATION ---
ASL_CITIZEN_ROOT = os.path.join(ROOT_DIR, 'data', 'raw', 'ASL_Citizen')
RAW_VIDEOS_DIR = os.path.join(ASL_CITIZEN_ROOT, 'videos')
SPLITS_DIR = os.path.join(ASL_CITIZEN_ROOT, 'splits')

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def get_next_sequence_number(action_path):
    if not os.path.exists(action_path):
        os.makedirs(action_path)
        return 0
    
    existing_folders = [int(f) for f in os.listdir(action_path) if f.isdigit()]
    if not existing_folders:
        return 0
    return max(existing_folders) + 1

def process_video(video_path, action, sequence_num):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return False

    if total_frames >= SEQUENCE_LENGTH:
        frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=int)
    else:
        frame_indices = np.arange(total_frames)
        if total_frames < 10: 
            cap.release()
            return False
            
    save_path = os.path.join(OUTPUT_PATH, action, str(sequence_num))
    os.makedirs(save_path, exist_ok=True)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        current_frame = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if total_frames < SEQUENCE_LENGTH or current_frame in frame_indices:
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                
                npy_path = os.path.join(save_path, str(saved_count))
                np.save(npy_path, keypoints)
                saved_count += 1
                
                if saved_count == SEQUENCE_LENGTH:
                    break
            
            current_frame += 1
            
        while saved_count < SEQUENCE_LENGTH:
            npy_path = os.path.join(save_path, str(saved_count))
            np.save(npy_path, keypoints) 
            saved_count += 1

    cap.release()
    return True

def main():
    print(f"🚀 Processing ASL Citizen Dataset...")
    print(f"🎯 Target Cap: {NO_SEQUENCES} videos per word")
    
    if not os.path.exists(SPLITS_DIR):
        print("❌ Splits folder not found.")
        return

    # 1. Load CSVs
    all_csvs = glob.glob(os.path.join(SPLITS_DIR, "*.csv"))
    if not all_csvs: return
        
    df_list = []
    for f in all_csvs:
        try: df_list.append(pd.read_csv(f))
        except: pass
    
    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)

    if 'Gloss' not in df.columns: return

    # 2. Filter
    target_actions = [a.lower() for a in ACTIONS]
    df['Gloss_Lower'] = df['Gloss'].astype(str).str.lower()
    filtered_df = df[df['Gloss_Lower'].isin(target_actions)]
    
    print(f"   Found {len(filtered_df)} candidate videos.")

    # 3. Process Loop
    for index, row in filtered_df.iterrows():
        gloss = row['Gloss_Lower']
        filename = row['Video file']
        
        # Identify Action
        action_name = next(a for a in ACTIONS if a.lower() == gloss)
        action_dir = os.path.join(OUTPUT_PATH, action_name)
        
        # --- CRITICAL FIX: CHECK LIMIT BEFORE PROCESSING ---
        current_count = get_next_sequence_number(action_dir)
        if current_count >= NO_SEQUENCES:
            # We don't print "Skipping" for every single file to keep logs clean
            # but we assume if one hits the limit, we don't need more of this word
            continue 

        video_path = os.path.join(RAW_VIDEOS_DIR, filename)
        if not os.path.exists(video_path):
            video_path += ".mp4"
            if not os.path.exists(video_path):
                continue
        
        print(f"   🎬 Processing '{action_name}' -> Sequence {current_count}...")
        success = process_video(video_path, action_name, current_count)
        
        if success:
            print(f"      ✅ Saved.")

    print("\n✅ Processing Complete.")
    print("   Note: If you didn't see logs for 'fever', it's because you already have enough data.")

if __name__ == "__main__":
    main()