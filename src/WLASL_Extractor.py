import json
import os
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
import requests
import sys
import time

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import SEQUENCE_LENGTH
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
except ImportError:
    SEQUENCE_LENGTH = 30
    DATA_PATH = os.path.join('data', 'processed')

# --- CONFIGURATION ---
# Put your JSON file name here (Make sure it is in the Signify root folder)
JSON_FILE = os.path.join(ROOT_DIR, 'WLASL_v0.3.json') 
TARGET_GLOSSES = ['fever', 'pain', 'medicine'] 

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def download_video(url, video_id):
    """
    Smart downloader that handles both YouTube and Direct MP4 links.
    """
    output_filename = f"temp_{video_id}.mp4"
    
    # 1. Handle YouTube
    if "youtube.com" in url or "youtu.be" in url:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': f'temp_{video_id}.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_filename
        except Exception as e:
            # print(f"   ⚠️ YouTube Download Failed: {e}")
            return None

    # 2. Handle Direct MP4 Links (handspeak, asldeafined, etc.)
    else:
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(output_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                return output_filename
        except Exception as e:
            # print(f"   ⚠️ Direct Download Failed: {e}")
            return None
    return None

def process_video_instance(video_path, label, instance_id, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle bad video reads
    if total_frames == 0:
        cap.release()
        return False

    # Define the range to extract based on JSON data
    # If end_frame is -1, it means "until the end"
    real_end = total_frames if end_frame == -1 else min(end_frame, total_frames)
    real_start = max(0, start_frame)
    
    # Validation: If the clip is too short or invalid
    if real_end <= real_start:
        cap.release()
        return False

    # Create folder: data/processed/pain/0, data/processed/pain/1, etc.
    # We use instance_id to ensure unique folders
    save_path = os.path.join(DATA_PATH, label, str(instance_id))
    os.makedirs(save_path, exist_ok=True)

    # Calculate exactly which frames to grab to get SEQUENCE_LENGTH (30) frames
    # We pick 30 frames evenly distributed between real_start and real_end
    frame_indices = np.linspace(real_start, real_end - 1, SEQUENCE_LENGTH, dtype=int)
    
    processed_count = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        current_frame = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process if this frame is in our list of frames to keep
            if current_frame in frame_indices:
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                
                npy_path = os.path.join(save_path, str(processed_count))
                np.save(npy_path, keypoints)
                processed_count += 1
                
                if processed_count >= SEQUENCE_LENGTH:
                    break
            
            current_frame += 1
            
    cap.release()
    
    # Cleanup: If we failed to extract enough frames, delete the folder
    if processed_count < SEQUENCE_LENGTH:
        import shutil
        shutil.rmtree(save_path)
        return False
        
    return True

def main():
    print(f"🚀 Loading WLASL JSON from: {JSON_FILE}")
    
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {JSON_FILE}")
        print("Please ensure the WLASL_v0.3.json file is in the Signify root folder.")
        return

    print(f"🎯 Target Glosses: {TARGET_GLOSSES}")

    # Loop through every word in the dictionary
    for entry in data:
        gloss = entry['gloss']
        
        # Check if this is a word we want
        if gloss in TARGET_GLOSSES:
            print(f"\n🎥 Found gloss: '{gloss}' - Processing instances...")
            
            instances = entry['instances']
            success_count = 0
            
            for i, inst in enumerate(instances):
                url = inst['url']
                video_id = inst['video_id']
                start_frame = inst['frame_start']
                end_frame = inst['frame_end']
                
                print(f"   [{i+1}/{len(instances)}] Downloading {url}...")
                
                # 1. Download
                temp_file = download_video(url, video_id)
                
                if temp_file and os.path.exists(temp_file):
                    # 2. Process
                    success = process_video_instance(temp_file, gloss, i, start_frame, end_frame)
                    
                    if success:
                        print(f"      ✅ Processed & Saved.")
                        success_count += 1
                    else:
                        print(f"      ⚠️ Processing Failed (Video empty or too short).")
                    
                    # 3. Cleanup temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                else:
                    print(f"      ❌ Download Failed (Link might be dead).")
            
            print(f"🏁 Finished '{gloss}'. Total valid sequences: {success_count}")

if __name__ == "__main__":
    main()