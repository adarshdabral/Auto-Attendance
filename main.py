import cv2
import face_recognition
import os
import csv
from datetime import datetime
from threading import Thread

# Load known faces and names
known_faces = []
known_names = []

# Function to load known faces from a folder
def load_known_faces(folder_path):
    for filename in os.listdir(folder_path):
        img = face_recognition.load_image_file(os.path.join(folder_path, filename))
        encoding = face_recognition.face_encodings(img)[0]
        known_faces.append(encoding)
        known_names.append(os.path.splitext(filename)[0])

# Load known faces from a folder (replace 'path_to_known_faces' with the actual path)
load_known_faces('./Faces')

# Initialize some variables
face_locations = []
face_encodings = []
present_students = set()

# Open the camera
cap = cv2.VideoCapture(0)

# Function to display loading animation
def show_loading_animation():
    animation_frames = ['|', '/', '-', '\\']
    frame_index = 0
    while not known_faces:
        loading_message = "Loading known faces. Please wait... " + animation_frames[frame_index]
        print(loading_message, end='\r')
        frame_index = (frame_index + 1) % len(animation_frames)
        cv2.waitKey(200)  # Wait for 200 milliseconds between frames
    print("\nKnown faces loaded!")

# Start a thread to display the loading animation
loading_thread = Thread(target=show_loading_animation)
loading_thread.start()

# Wait for the loading thread to finish before proceeding
loading_thread.join()

# Main loop for face recognition
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            present_students.add(name)

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Attendance System', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Save attendance records to a CSV file with date and time in the filename
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        csv_file_path = f'attendance_list_{current_datetime}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['StudentName']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write each student's name
            for student in present_students:
                writer.writerow({'StudentName': student})
        print(f"Attendance list saved to: {csv_file_path}")
        break  # Break the main loop when 'q' is pressed

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
