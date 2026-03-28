import os

print("Step 1: Preprocessing...")
os.system("python src/preprocess.py")

print("Step 2: Training...")
os.system("python src/train.py")

print("Step 3: Starting Detection...")
os.system("python src/detect.py")