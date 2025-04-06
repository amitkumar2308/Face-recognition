import cv2
import numpy as np
import os

# Haarcascade model load 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

FACE_DIR = "faces/"

def train_faces():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(FACE_DIR):
        person_path = os.path.join(FACE_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        
        label_map[current_label] = person_name  # Naam ka label create karo

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Face detect aur crop karo
            detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in detected_faces:
                face = gray[y:y+h, x:x+w]  # Sirf face crop karo
                faces.append(face)
                labels.append(current_label)
        
        current_label += 1

    recognizer.train(faces, np.array(labels))  # Model train karo
    recognizer.save("face_model.yml")  # Model save karna
    np.save("label_map.npy", label_map)  # Naam mapping save karna
    print("✅ Training complete!")

train_faces()
