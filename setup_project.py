import os

def create_files():
    # 1. Create Folder Structure
    folders = [
        "data/processed",
        "models",
        "logs",
        "src",
        "web_app/public/model",
        "web_app/src/components"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # 2. Create requirements.txt (Dependencies)
    requirements = """tensorflow==2.15.0
numpy<2.0.0
opencv-python
mediapipe
scikit-learn
matplotlib
python-dotenv
tensorflowjs
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("Created requirements.txt")

    # 3. Create config.py (Settings)
    config_code = """import os
import numpy as np

DATA_PATH = os.path.join('data', 'processed') 
ACTIONS = np.array(['fever', 'pain', 'medicine']) 
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
"""
    with open("src/config.py", "w") as f:
        f.write(config_code)
    print("Created src/config.py")

    # 4. Create empty data collection file
    with open("src/data_collection.py", "w") as f:
        f.write("# Code will be added here later")
    print("Created src/data_collection.py")

    print("\nSUCCESS: Project structure is ready.")

if __name__ == "__main__":
    create_files()