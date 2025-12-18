import numpy as np
import os

# 1. Pick a specific file to check (Adjust path if needed)
# Example: Checking the 0th frame of the 0th video for 'fever'
file_path = os.path.join('data', 'processed', 'fever', '0', '0.npy')

if os.path.exists(file_path):
    # 2. Load the binary data
    data = np.load(file_path)
    
    print(f"✅ Success! File found: {file_path}")
    print(f"📊 Shape of data: {data.shape}")
    print(f"🔢 First 10 numbers: {data[:10]}")
    print("---------------------------------")
    print("If you see numbers above, your data collection worked!")
else:
    print(f"❌ File not found at: {file_path}")
    print("Check your folders in 'data/processed' to see what actions you recorded.")