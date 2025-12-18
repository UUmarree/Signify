import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

# --- ROBUST IMPORT BLOCK ---
try:
    from src.config import ACTIONS, SEQUENCE_LENGTH
except ImportError:
    try:
        from config import ACTIONS, SEQUENCE_LENGTH
    except ImportError:
        print("❌ Config not found. Please ensure 'src/config.py' exists.")
        sys.exit(1)

MODELS_PATH = os.path.join(ROOT_DIR, 'models')

import mediapipe as mp

# --- LOAD MODEL ---
model_path = os.path.join(MODELS_PATH, 'action.h5')
print(f"🧠 Loading model from: {model_path}")

if not os.path.exists(model_path):
    print("❌ Model not found. Did you run train_model.py?")
    sys.exit(1)

model = load_model(model_path)
print("✅ Model loaded successfully!")

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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8 

    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # --- HAND GATE LOGIC ---
            # Check if either hand is present in the frame
            hands_present = (results.left_hand_landmarks or results.right_hand_landmarks)

            if hands_present:
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]
                
                if len(sequence) == SEQUENCE_LENGTH:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    predictions = predictions[-20:]
                    
                    # Logic: Wait for 15 stable frames (approx 0.5s)
                    if np.unique(predictions[-15:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            if len(sentence) > 0: 
                                if ACTIONS[np.argmax(res)] != sentence[-1]:
                                    sentence.append(ACTIONS[np.argmax(res)])
                            else:
                                sentence.append(ACTIONS[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
            else:
                # If hands are missing, reset the sequence to avoid using old frames
                # This prevents the model from "remembering" the last sign forever
                if len(sequence) > 0:
                    sequence = []
                    predictions = []
                
                # Optional: Show user they need to put hands up
                cv2.putText(image, 'WAITING FOR HANDS...', (200, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Removed prob_viz() function call here
            
            # Only draw the Sentence Box at the top
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()