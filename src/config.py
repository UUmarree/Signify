import os
import numpy as np

# --- PATHS ---
# Robust path logic to ensure it works from notebooks, src, or root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed') 
MODELS_PATH = os.path.join(ROOT_DIR, 'models')

# --- MODEL SETTINGS ---
# The "Triage 10" - A complete medical scenario vocabulary
# 'nothing' is CRITICAL for preventing false positives when the user is idle.
ACTIONS = np.array([
     
    'pain', 
    'medicine', 
    'hello', 
    'thankyou', 
    'yes', 
    'no', 
    'help', 
    'blood', 
    'water',
    'nothing'
])

# Data collection parameters
NO_SEQUENCES = 50   # Number of videos per word
SEQUENCE_LENGTH = 30 # Frames per video (1 second)