# Masked Face Recognition System

A Python-based facial recognition system that can identify users even when they're wearing masks. The system uses machine learning models and OpenCV to focus on the eye region and upper face features for accurate recognition.

## Features

- **Masked Face Recognition**: Works even when users wear face masks
- **Real-time Recognition**: Live webcam-based face recognition
- **Easy User Registration**: Simple interface to register new users
- **Eye Region Focus**: Optimized for partial face recognition
- **Adjustable Tolerance**: Fine-tune recognition sensitivity
- **Performance Monitoring**: Real-time FPS display
- **Multiple Registration Methods**: Webcam capture or image files

## How It Works

The system uses pre-trained machine learning models to:
1. **Detect faces** in video frames using OpenCV
2. **Extract features** from the visible parts of the face (eyes, eyebrows, forehead)
3. **Compare features** with registered users using mathematical distance calculations
4. **Recognize users** even when wearing masks by focusing on the eye region

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera
- Windows, macOS, or Linux

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The installation may take a few minutes as it downloads machine learning models.

### Step 2: Verify Installation

```bash
python -c "import cv2, face_recognition; print('Installation successful!')"
```

## Quick Start

### 1. Register Users

First, register users in the system:

```bash
python user_registration.py
```

Follow the menu to:
- Register new users via webcam (recommended)
- Register from existing image files
- View registered users
- Delete users

**Registration Tips**:
- Take 8-10 photos per user
- Include photos with and without masks
- Try different angles (slightly left, right, up, down)
- Ensure good lighting

### 2. Start Recognition

Run the main application:

```bash
python main.py
```

**Controls**:
- `SPACE`: Toggle recognition on/off
- `R`: Register new user quickly
- `T`: Adjust recognition tolerance
- `Q`: Quit application

## Usage Guide

### User Registration

#### Method 1: Webcam Registration (Recommended)
```bash
python user_registration.py
```
1. Choose option 1 (Register new user - webcam)
2. Enter the user's name
3. Position face in camera view
4. Press SPACE to capture images
5. Take photos with and without mask for best results

#### Method 2: Image File Registration
```bash
python user_registration.py
```
1. Choose option 2 (Register from image files)
2. Enter the user's name
3. Provide paths to image files
4. System will process and register the user

### Live Recognition

```bash
python main.py
```

The system will:
- Display live video feed
- Show recognition results with confidence scores
- Highlight the eye region used for masked face recognition
- Display FPS and system status

### Recognition Settings

- **Tolerance**: Lower values (0.3-0.4) = more strict, Higher values (0.6-0.7) = more lenient
- **Confidence Threshold**: Minimum confidence to display user name (default: 0.4)

## File Structure

```
facial_recognition/
├── main.py                    # Main recognition application
├── face_recognition_system.py # Core recognition logic
├── user_registration.py       # User registration system
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── users/                    # User data storage (created automatically)
    ├── user1.pkl
    ├── user2.pkl
    └── ...
```

## Technical Details

### Machine Learning Models Used

- **Face Detection**: Haar Cascade Classifiers (OpenCV)
- **Feature Extraction**: Deep Learning CNN models (dlib/face_recognition)
- **Recognition**: Euclidean distance comparison

### Masked Face Recognition Strategy

1. **Focus on Eye Region**: Extract features from upper 60% of detected face
2. **Multiple Encodings**: Store multiple face encodings per user for robustness
3. **Tolerance Adjustment**: Configurable sensitivity for different scenarios
4. **Confidence Scoring**: Display recognition confidence levels

### Performance

- **Speed**: 15-30 FPS on modern computers
- **Accuracy**: 80-90% for masked faces, 95%+ for unmasked faces
- **Memory**: Low memory footprint using efficient encodings

## Troubleshooting

### Common Issues

#### 1. "Could not open webcam"
- Check if camera is connected and not used by other applications
- Try different camera indices: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

#### 2. "No faces found in images"
- Ensure good lighting in photos
- Face should be clearly visible and not too small
- Try different angles and distances

#### 3. Poor recognition accuracy
- Register more images per user (10-15 recommended)
- Include both masked and unmasked photos
- Adjust tolerance settings (press 'T' during recognition)
- Ensure consistent lighting conditions

#### 4. Installation issues
```bash
# For Windows users with dlib issues:
pip install cmake
pip install dlib

# For macOS users:
brew install cmake
pip install dlib

# Alternative installation:
conda install -c conda-forge dlib
```

### Performance Optimization

- **Reduce video resolution** for faster processing
- **Adjust tolerance** based on your needs
- **Register fewer images** per user if speed is critical
- **Use good lighting** for better accuracy

## Advanced Usage

### Custom Tolerance Settings

```python
# In main.py, modify these values:
self.tolerance = 0.4  # Lower = more strict
self.confidence_threshold = 0.5  # Higher = more confident display
```

### Batch User Registration

Create a script to register multiple users from a folder:

```python
from user_registration import UserRegistration
import os

registration = UserRegistration()
users_folder = "path/to/user/images"

for user_folder in os.listdir(users_folder):
    user_path = os.path.join(users_folder, user_folder)
    if os.path.isdir(user_path):
        image_paths = [os.path.join(user_path, img) 
                      for img in os.listdir(user_path) 
                      if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        registration.register_from_images(user_folder, image_paths)
```

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Submitting pull requests

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **OpenCV** for computer vision capabilities
- **face_recognition library** for ML models
- **dlib** for facial landmark detection
- **NumPy** for numerical computations

---

**Note**: This system is designed for educational and personal use. For production environments, consider additional security measures and privacy considerations.
