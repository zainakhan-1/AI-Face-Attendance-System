import cv2
import numpy as np
import ast
from datetime import datetime

def mark_attendance(name):
    with open("attendance.csv", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{now}\n")

def recognize_and_mark():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    with open("labels.txt") as f:
        label_map = ast.literal_eval(f.read())
    id_map = {v: k for k, v in label_map.items()}

    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    marked = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            name = id_map.get(id_, "Unknown")
            if conf < 70:
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if name not in marked:
                    mark_attendance(name)
                    marked.add(name)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Recognizing", frame)

        # 🚨 Press ESC or if the window is closed manually
        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty("Recognizing", cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()
