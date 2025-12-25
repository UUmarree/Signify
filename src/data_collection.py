import cv2
import numpy as np
import os
import mediapipe as mp
import sys

# --- PATH SETUP ---
# This forces the script to find the 'data' folder in the PROJECT ROOT, 
# no matter where you run this script from.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

# Import config
try:
    from src.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH
    # We override DATA_PATH to ensure it uses the absolute path we just calculated
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed')
except ImportError as e:
    print(f"❌ Error importing config: {e}")
    print("Make sure src/config.py exists.")
    sys.exit(1)

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

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# --- HELPER: Find next available folder ---
def get_next_sequence(action_path):
    if not os.path.exists(action_path):
        return 0
    # List all folders that are numbers
    dirs = [int(f) for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f)) and f.isdigit()]
    if not dirs:
        return 0
    return max(dirs) + 1

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Smart Data Collection")
    print(f"📂 Saving data to Absolute Path: {DATA_PATH}")
    print(f"🎯 Target: {NO_SEQUENCES} sequences per word")

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        sys.exit(1)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in ACTIONS:
            action_path = os.path.join(DATA_PATH, action)
            
            # 1. Check how many we already have
            start_sequence = get_next_sequence(action_path)
            
            # 2. Logic: Do we need more?
            if start_sequence >= NO_SEQUENCES:
                print(f"✅ SKIPPING: '{action}' already has {start_sequence} sequences.")
                continue
            
            needed = NO_SEQUENCES - start_sequence
            print(f"🎥 RECORDING: '{action}' | Existing: {start_sequence} | Need: {needed}")

            # 3. Create folder if it doesn't exist
            try: 
                os.makedirs(action_path, exist_ok=True)
            except Exception as e:
                print(f"❌ CRITICAL ERROR: Could not create folder {action_path}")
                sys.exit(1)

            # 4. Loop from the NEXT available number
            for sequence in range(start_sequence, NO_SEQUENCES):
                
                # Create the specific sequence folder (e.g. fever/30)
                try: 
                    os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)
                except:
                    pass

                for frame_num in range(SEQUENCE_LENGTH):

                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Error: Lost webcam feed.")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    # UI LOGIC
                    if frame_num == 0: 
                        cv2.putText(image, 'GET READY', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting: {action} | Video #{sequence}', (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000) # 2 Second pause
                    else: 
                        cv2.putText(image, f'Collecting: {action} | Video #{sequence}', (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # SAVE LOGIC (With Error Checking)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(action_path, str(sequence), str(frame_num))
                    try:
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        print(f"❌ Error saving frame: {e}")

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit()
                
                print(f"   ✅ Saved Sequence {sequence} for {action}")

    cap.release()
    cv2.destroyAllWindows()
    print("🎉 All missing data collected!")