import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Paths
TRAINING_DATA_DIR = "training_data"

# Initialize Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

app = Flask(__name__)

def compare_histograms(img1, img2):
    """
    Compare two images using histogram correlation.
    """
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation

# 

def capture_face():
    """
    Capture a face and return the cropped face image (headless mode compatible).
    """
    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)

    video_capture = cv2.VideoCapture(0)
    captured_face = None
    print("Press 'c' to capture a face or 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            captured_face = gray[y:y + h, x:x + w]

        # Use Matplotlib instead of cv2.imshow
        if captured_face is not None:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        key = input("Press 'c' to capture or 'q' to quit: ").strip().lower()
        if key == 'c' and captured_face is not None:  
            print("Face captured.")
            video_capture.release()
            return captured_face
        elif key == 'q':  
            print("Exiting...")
            video_capture.release()
            return None


def save_new_face(face, user_name):
    """
    Save the new face with a given name.
    """
    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)

    file_path = os.path.join(TRAINING_DATA_DIR, f"{user_name}.jpg")
    cv2.imwrite(file_path, face)
    print(f"New user {user_name} saved at {file_path}.")

def match_face(captured_face):
    """
    Match the captured face against known faces.
    """
    if not os.path.exists(TRAINING_DATA_DIR):
        print("No training data found.")
        return None, None

    known_faces = []
    known_labels = []

    for file_name in os.listdir(TRAINING_DATA_DIR):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(TRAINING_DATA_DIR, file_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            label = file_name.split(".")[0]
            known_faces.append(image)
            known_labels.append(label)

    captured_face_resized = cv2.resize(captured_face, (100, 100))

    best_match = None
    best_score = -1  # Correlation is higher when closer to 1
    for idx, known_face in enumerate(known_faces):
        known_face_resized = cv2.resize(known_face, (100, 100))
        score = compare_histograms(captured_face_resized, known_face_resized)
        if score > best_score:
            best_score = score
            best_match = known_labels[idx]

    return best_match, best_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    action = request.form['action']
    if action == 'register':
        face = capture_face()
        if face is not None:
            user_name = request.form['username']
            save_new_face(face, user_name)
            return render_template('index.html', message=f"User {user_name} registered successfully.")
        return render_template('index.html', message="Failed to capture face.")
    elif action == 'recognize':
        face = capture_face()
        if face is not None:
            best_match, score = match_face(face)
            if best_match and score > 0.5:
                return render_template('index.html', message=f"Match Found: {best_match}")
            else:
                return render_template('index.html', message="No Match Found!")
        return render_template('index.html', message="Failed to capture face.")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
