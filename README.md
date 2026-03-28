Emotion Detection from Face 

Project Overview:

This project detects human emotions in real-time using facial expressions. It uses a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
Dataset download link: https://www.kaggle.com/datasets/xavier00/fer2013-facial-expression-recognition-dataset?resource=download

Features:

-Real-time face detection using OpenCV
-Emotion classification (7 classes)
-CNN-based deep learning model
-Fully executable via command line

Emotions Detected:

-Angry
-Disgust
-Fear
-Happy
-Sad
-Surprise
-Neutral

Installation:
pip install -r requirements.txt

Command Line Execution:

Step 1: Preprocessing
python src/preprocess.py

Step 2: Training
python src/train.py --epochs 10 --batch_size 64

Step 3: Real-Time Detection
python src/detect.py --camera 0

Run Complete Pipeline
python run.py

Output:
Accuracy Graph
<img width="955" height="779" alt="accuracy" src="https://github.com/user-attachments/assets/95ab8f3c-d2ba-4966-9939-9139f4e59b78" />


Detection Results
[Output1] <img width="845" height="669" alt="output1" src="https://github.com/user-attachments/assets/af751ec6-cb6e-4d1e-b808-99f2d4d0f0c2" />
[Output2] <img width="837" height="662" alt="output2" src="https://github.com/user-attachments/assets/1e50cad5-ee98-4632-9d2b-9ee2be755d60" />


Technologies Used:

-Python
-TensorFlow / Keras
-OpenCV
-NumPy, Pandas

Notes:

-Model file is included in `/model`
-If missing, run training script to generate it

Conclusion:
The system successfully detects human emotions in real-time using deep learning and computer vision techniques.


Author:
Dilisha Khan
