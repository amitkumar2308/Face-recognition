import cv2
import numpy as np

# Haarcascade load karo
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Trained model load karo
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

# Label mapping load karo
label_map = np.load("label_map.npy", allow_pickle=True).item()

def recognize_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in detected_faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)

            name = label_map.get(label, "Unknown")  # Agar label match nahi hota to "Unknown" dikhao
            confidence_text = f"{100 - confidence:.2f}%"  # Confidence ko percentage me show karo

            color = (0, 255, 0) if confidence < 50 else (0, 0, 255)  # Confidence ke basis pe color change
            cv2.putText(frame, f"{name} ({confidence_text})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize_faces()
