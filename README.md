# Face Recognition System Using Histogram Matching

## Overview
This project is a simple face recognition system implemented using OpenCV and Flask. It captures and stores user faces, then recognizes them using histogram matching based on correlation.

## Features
- Face detection using OpenCV's Haar cascade classifier.
- Face registration: Capture and store user faces.
- Face recognition: Match a captured face against stored images using histogram comparison.
- Web interface built with Flask for easy interaction.

## Technologies Used
- Python
- OpenCV
- Flask
- Matplotlib
- NumPy

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/abbhiii/Face_Recognition.git
cd Face_Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

Then, open `http://127.0.0.1:5000/` in your browser.

## Usage
### 1. Register a New User
- Enter a username and capture a face.
- The captured face is saved in the `training_data` directory.

### 2. Recognize a Face
- Capture a new face.
- The system compares it with stored faces using histogram correlation.
- If a match is found (correlation > 0.5), it displays the user's name.


## Limitations
- Histogram matching is sensitive to lighting conditions.
- It does not consider facial features beyond pixel intensity distributions.
- Works best with controlled lighting and front-facing images.

## Future Enhancements
- Replace histogram matching with deep learning-based face recognition (e.g., FaceNet, OpenFace).
- Improve the web interface with better UI.
- Add a database for better user management.


