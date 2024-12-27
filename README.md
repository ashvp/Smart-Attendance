# Facial Recognition-Based Attendance System

## Overview
The **Facial Recognition-Based Attendance System** is a state-of-the-art solution designed to modernize attendance management by leveraging real-time facial recognition and spoof detection. This system ensures secure, accurate, and efficient attendance tracking, minimizing manual errors and enhancing operational efficiency. 

---

## Features

### Core Features
- **Automated Attendance Tracking**:
  - Real-time facial recognition with high accuracy.
  - Automatic attendance marking with timestamps.

- **Advanced Spoof Detection**:
  - Robust CNN-based model to distinguish real faces from spoofed images or videos.

- **User Management**:
  - Easy registration process with multiple image captures for robust encoding.

- **Streamlined UI**:
  - Intuitive interface powered by Streamlit for seamless user interaction.

- **Secure and Reliable**:
  - All user data and attendance records are securely stored in a SQLite database.

---

## Technologies Used

- **Programming Language**: Python
- **Frontend**: Streamlit
- **Core Libraries**:
  - `face_recognition` for face detection and encoding.
  - `tensorflow.keras` for spoof detection.
  - `opencv-python` for video and image processing.
  - `sqlite3` for database management.
- **Hardware Requirements**: Standard webcam for image capture.

---

## Installation

### Prerequisites
- Python 3.8 or later
- Installed libraries:
  ```bash
  pip install streamlit face_recognition opencv-python tensorflow
  ```

- **Spoof Detection Model**: Download the pre-trained `spoofPredictionModel.h5` and place it in the project directory.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/facial-recognition-attendance.git
   cd facial-recognition-attendance
   ```
2. Initialize the database:
   ```bash
   python initialize_db.py
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## How to Use

### Add New Student
1. Select "Add New Student" from the dropdown menu.
2. Enter the student's name and capture multiple images using the webcam.
3. Save the encoding to the database.

### Mark Attendance
1. Select "Mark Attendance" from the dropdown menu.
2. Allow the webcam to capture your face.
3. The system will verify your identity and mark attendance if the face is recognized and verified as real.

---

## Future Enhancements

- **Cloud Integration**:
  - Enable cloud-based storage for centralized data access.

- **Mobile Application**:
  - Develop a mobile app for attendance tracking on the go.

- **Advanced Security Features**:
  - Integrate liveness detection and multi-factor authentication.

- **Data Analytics**:
  - Provide insights into attendance trends and patterns.

- **Support for Edge Devices**:
  - Optimize for deployment on low-power devices.

---

## Contribution Guidelines

We welcome contributions to enhance this project! Follow these steps to contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
