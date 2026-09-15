import cv2
import numpy as np
from PIL import Image
import os

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def get_images_and_labels(path="dataset"):
        image_paths = []
        labels = []
        label_map = {}
        current_label = 0
        for name in os.listdir(path):
            person_folder = os.path.join(path, name)
            if not os.path.isdir(person_folder):
                continue
            for img_file in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_file)
                img = Image.open(img_path).convert('L')
                np_img = np.array(img, 'uint8')
                faces = detector.detectMultiScale(np_img)
                for (x, y, w, h) in faces:
                    image_paths.append(np_img[y:y+h, x:x+w])
                    if name not in label_map:
                        label_map[name] = current_label
                        current_label += 1
                    labels.append(label_map[name])
        return image_paths, labels, label_map

    faces, ids, label_map = get_images_and_labels()
    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer.yml")
    with open("labels.txt", "w") as f:
        f.write(str(label_map))
