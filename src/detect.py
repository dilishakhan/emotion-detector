import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)

args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)

model = load_model("model/emotion_model.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = face.reshape(1,48,48,1)

        pred = model.predict(face)
        label = emotion_labels[np.argmax(pred)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()