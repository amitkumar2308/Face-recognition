import cv2
import os

# Folder jisme images store hongi
FACE_DIR = "faces/"
os.makedirs(FACE_DIR, exist_ok=True)

def capture_images(name):
    cap = cv2.VideoCapture(0)
    count = 0
    user_folder = os.path.join(FACE_DIR, name)
    os.makedirs(user_folder, exist_ok=True)  # Folder create karega

    while count < 150: 
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        face_path = f"{user_folder}/{count}.jpg"
        cv2.imwrite(face_path, frame)  # Pure image save karega bina crop kiye

        cv2.imshow("Capturing Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ {count} Images saved for {name}")

#User ka naam input lo
user_name = input("Enter your name: ")
capture_images(user_name)
