import streamlit as st
import cv2
import os
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# SQLite Database Setup
DB_NAME = "attendance_system.db"

# Load the spoof detection model
spoof_model = load_model("spoofPredictionModel.h5")

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create Students table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        encoding BLOB
    )
    ''')
    # Create Attendance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        timestamp TEXT,
        FOREIGN KEY(student_id) REFERENCES students(id)
    )
    ''')
    conn.commit()
    conn.close()

# Save the face encoding to the database
def save_encoding(name, encoding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (name, encoding) VALUES (?, ?)", 
                   (name, sqlite3.Binary(encoding.tobytes())))
    conn.commit()
    conn.close()

# Load all face encodings from the database
def load_encodings():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM students")
    rows = cursor.fetchall()
    student_encodings = {}
    for row in rows:
        student_encodings[row[0]] = np.frombuffer(row[1], dtype=np.float64)
    conn.close()
    return student_encodings

# Mark attendance for a student
def mark_attendance(name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM students WHERE name = ?", (name,))
    student_id = cursor.fetchone()

    if student_id:
        student_id = student_id[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO attendance (student_id, timestamp) VALUES (?, ?)", 
                       (student_id, timestamp))
        conn.commit()
        st.write(f"Attendance for {name} marked at {timestamp}")
    else:
        st.write("Student not found in the database.")

    conn.close()

# Add new student with face encoding
def add_new_student(name):
    st.write(f"Capturing images for new student: {name}")
    cap = cv2.VideoCapture(0)
    images = []
    while len(images) < 5:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            continue
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            images.append(frame)
            st.write(f"Captured image {len(images)}")
    cap.release()
    cv2.destroyAllWindows()

    # Get encodings for the captured images
    encodings = []
    for img in images:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_img)
        encodings.extend(face_recognition.face_encodings(rgb_img, boxes))

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)  # Take the average encoding from multiple pictures
        save_encoding(name, avg_encoding)
        st.write(f"Student {name} added successfully!")
    else:
        st.write("No face detected. Please try again.")

# Check if the face is real or spoof
def is_real_face(frame):
    """
    Predicts if a face is real or spoof using the loaded CNN model.
    Ensures the input shape matches the model's requirements.
    """
    model_input_size = spoof_model.input_shape[1:3]  # Get expected input dimensions (e.g., 224x224)
    resized_frame = cv2.resize(frame, model_input_size)  # Resize the frame
    img_array = img_to_array(resized_frame)
    img_array = preprocess_input(img_array)  # Normalize as required by the model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict and return whether the face is real or spoof
    prediction = spoof_model.predict(img_array)[0][0]  # Get the prediction score
    st.write(f"Spoof detection score: {prediction}")
    return prediction < 0.5  # Adjust threshold based on your model

# Mark attendance with spoof detection
def mark_attendance_with_spoof_check():
    student_encodings = load_encodings()
    st.write("Marking attendance")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.write("Failed to capture image.")
        return

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img, boxes)

    for encoding in encodings:
        # Run spoof detection
        st.write("Checking if face is real or spoof.")
        if not is_real_face(frame):
            st.write("Spoof detected! Attendance not marked.")
            continue

        # Check if the face matches a known encoding
        matches = face_recognition.compare_faces(list(student_encodings.values()), encoding, tolerance=0.6)
        if True in matches:
            match_index = matches.index(True)
            name = list(student_encodings.keys())[match_index]
            st.write(f"Face recognized as {name}.")
            mark_attendance(name)
        else:
            st.write("Real face detected, but not recognized in the database.")

# Streamlit frontend
def main():
    st.title("Facial Recognition Attendance System")
    init_db()  # Initialize the database if not already initialized

    option = st.selectbox("Choose an option", ["Mark Attendance", "Add New Student", "Exit"])

    if option == "Mark Attendance":
        mark_attendance_with_spoof_check()

    elif option == "Add New Student":
        name = st.text_input("Enter student name")
        if st.button("Add Student"):
            if name:
                add_new_student(name)
            else:
                st.write("Please enter a name.")

    elif option == "Exit":
        st.write("Exiting the application")

if __name__ == "__main__":
    main()
