import cv2
import numpy as np
import os
import sys
import time

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(ROOT_DIR)

try:
    from src.config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH
except ImportError:
    print("❌ Config not found.")
    sys.exit(1)

# --- DRAWING UTILITIES ---
def draw_landmarks(image, keypoints):
    """
    Reverse engineers the flattened keypoints back into (x,y) coordinates
    and draws them on a black canvas.
    """
    h, w, c = image.shape
    
    # Indices based on the concatenation order in data_collection.py:
    # Pose(132) + Face(1404) + LH(63) + RH(63)
    
    # 1. Extract Right Hand (Last 63 values -> 21 points * 3 dims)
    rh_end = len(keypoints)
    rh_start = rh_end - 63
    rh = keypoints[rh_start:rh_end].reshape(21, 3)
    
    # 2. Extract Left Hand (Before RH -> 63 values)
    lh_end = rh_start
    lh_start = lh_end - 63
    lh = keypoints[lh_start:lh_end].reshape(21, 3)
    
    # 3. Extract Pose (First 132 values -> 33 points * 4 dims [x,y,z,vis])
    pose = keypoints[0:132].reshape(33, 4)

    # Helper to draw points
    def draw_points(landmarks, color):
        for res in landmarks:
            cx, cy = int(res[0] * w), int(res[1] * h)
            # Only draw if valid coordinates
            if cx > 0 and cy > 0:
                cv2.circle(image, (cx, cy), 3, color, -1)

    # Draw Pose (White)
    draw_points(pose[:, :3], (200, 200, 200))
    
    # Draw Left Hand (Purple) - Note: If max value is 0, this loop does nothing visible
    if np.max(lh) > 0:
        draw_points(lh, (255, 0, 255))
        
    # Draw Right Hand (Orange)
    if np.max(rh) > 0:
        draw_points(rh, (0, 165, 255))

# --- MAIN PLAYER ---
def play_sequence(action, sequence_num):
    sequence_path = os.path.join(DATA_PATH, action, str(sequence_num))
    
    if not os.path.exists(sequence_path):
        print(f"❌ Sequence {sequence_num} not found for {action}")
        return

    print(f"🎬 Playing: {action.upper()} | Sequence #{sequence_num}")
    
    # Create a black window
    window_name = 'Data Visualization'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    for i in range(SEQUENCE_LENGTH):
        npy_path = os.path.join(sequence_path, f"{i}.npy")
        
        if not os.path.exists(npy_path):
            continue
            
        keypoints = np.load(npy_path)
        
        # Black background
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw info
        cv2.putText(image, f"Action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"Frame: {i}/{SEQUENCE_LENGTH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw Skeleton
        draw_landmarks(image, keypoints)
        
        cv2.imshow(window_name, image)
        
        # Play at roughly 30 FPS
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    # Pause briefly after video ends
    time.sleep(0.5)

if __name__ == "__main__":
    print("👀 Starting Data Visualizer...")
    print("Press 'q' inside the window to skip a video.")
    print("Press Ctrl+C in terminal to stop.")
    
    # Play 5 random videos from each class
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path): continue
        
        # Get list of sequences
        seqs = [f for f in os.listdir(action_path) if f.isdigit()]
        
        # Pick 3 random ones to check quality
        import random
        selected_seqs = random.sample(seqs, min(len(seqs), 3))
        
        for seq in selected_seqs:
            play_sequence(action, seq)

    cv2.destroyAllWindows()