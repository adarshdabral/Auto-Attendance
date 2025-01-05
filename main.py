import cv2
import os
import numpy as np
import csv
from datetime import datetime

# Path to the folder containing known face images
faces_folder = "faces"

# CSV file to log attendance
attendance_file = "attendance.csv"

# Load known faces and their names
known_faces = []
known_names = []

print("Loading known faces...")
for file_name in os.listdir(faces_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(faces_folder, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        known_faces.append(image)
        known_names.append(os.path.splitext(file_name)[0])

print(f"Loaded {len(known_faces)} known faces.")

# Initialize attendance log
attendance_log = set()

# Function to mark attendance
def mark_attendance(name):
    if name not in attendance_log:
        attendance_log.add(name)
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(attendance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, time_stamp])
        print(f"Attendance marked for {name} at {time_stamp}")

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Press 'q' to quit the program.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = gray_frame[y:y + h, x:x + w]
        face_region = cv2.resize(face_region, (100, 100))  # Resize to match known faces

        # Compare with known faces
        for i, known_face in enumerate(known_faces):
            # Use template matching or SSIM for comparison
            result = cv2.matchTemplate(face_region, known_face, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Threshold for a match (adjust as needed)
            if max_val > 0.6:
                name = known_names[i]
                mark_attendance(name)

                # Draw a rectangle and label around the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                break

    # Display the resulting frame
    cv2.imshow("Attendance System", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
