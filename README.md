# 👨‍💻 Modern Face Recognition Attendance System

A sleek, web-based application built with **Flask** and **OpenCV** to automate attendance tracking using face recognition. This project provides a robust, modern, and elegant UI for managing users and monitoring daily attendance records.
<img width="1886" height="822" alt="image" src="https://github.com/user-attachments/assets/eb8eb49d-f9bd-430f-be3d-fd2de6dd7219" />
<img width="1890" height="822" alt="image" src="https://github.com/user-attachments/assets/b52de793-c851-4099-990e-0c55cd1fb360" />

## ✨ Features

* **Real-time Face Detection & Recognition:** Utilizes OpenCV for capturing video feed and a trained K-Nearest Neighbors (K-NN) model for identifying individuals.
* **Modern & Elegant UI:** Built with **Bootstrap 5** for a responsive, clean, and user-friendly interface.
* **Simple User Management:** Easy registration of new users (Name & ID) by capturing face samples.
* **Attendance Logging:** Automatically logs the name, ID, and time of attendance to a daily CSV file.
* **Model Training:** Retrains the K-NN model dynamically whenever a new user is added or an existing user is deleted.
* **Persistent Storage:** Stores face data and daily attendance logs in organized local directories.

## ⚙️ Prerequisites

To run this project, you need to have **Python 3.x** installed, along with the following libraries:

```bash
pip install Flask opencv-python numpy scikit-learn pandas joblib
