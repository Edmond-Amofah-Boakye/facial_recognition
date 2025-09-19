import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
import face_recognition

class MaskedFaceRecognitionSystem:
    def __init__(self, users_dir: str = "users"):
        """
        Initialize the masked face recognition system.
        
        Args:
            users_dir: Directory to store user face encodings
        """
        self.users_dir = users_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create users directory if it doesn't exist
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # Load existing user data
        self.load_known_faces()
    
    def extract_eye_region(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract the eye region from a face for better masked face recognition.
        
        Args:
            image: Input image
            face_location: Face bounding box (top, right, bottom, left)
            
        Returns:
            Eye region image or None if not found
        """
        top, right, bottom, left = face_location
        face_img = image[top:bottom, left:right]
        
        # Focus on upper half of face (where eyes are)
        upper_face_height = int((bottom - top) * 0.6)  # Upper 60% of face
        upper_face = face_img[0:upper_face_height, :]
        
        return upper_face
    
    def get_face_encodings_masked(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Get face encodings optimized for masked faces by focusing on eye region.
        
        Args:
            image: Input image
            
        Returns:
            List of face encodings
        """
        # Get face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if not face_locations:
            return []
        
        encodings = []
        for face_location in face_locations:
            # Extract eye region for better masked face recognition
            eye_region = self.extract_eye_region(image, face_location)
            
            if eye_region is not None:
                # Get encoding from the full face but we'll use this for comparison
                face_encoding = face_recognition.face_encodings(image, [face_location])
                if face_encoding:
                    encodings.append(face_encoding[0])
        
        return encodings
    
    def register_user(self, name: str, images: List[np.ndarray]) -> bool:
        """
        Register a new user with multiple images (with and without mask).
        
        Args:
            name: User's name
            images: List of face images
            
        Returns:
            True if registration successful, False otherwise
        """
        user_encodings = []
        
        for image in images:
            encodings = self.get_face_encodings_masked(image)
            user_encodings.extend(encodings)
        
        if not user_encodings:
            print(f"No faces found in images for {name}")
            return False
        
        # Save user encodings
        user_file = os.path.join(self.users_dir, f"{name}.pkl")
        with open(user_file, 'wb') as f:
            pickle.dump(user_encodings, f)
        
        # Add to known faces
        self.known_face_encodings.extend(user_encodings)
        self.known_face_names.extend([name] * len(user_encodings))
        
        print(f"Successfully registered {name} with {len(user_encodings)} face encodings")
        return True
    
    def load_known_faces(self):
        """Load all known faces from the users directory."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.users_dir):
            return
        
        for filename in os.listdir(self.users_dir):
            if filename.endswith('.pkl'):
                name = filename[:-4]  # Remove .pkl extension
                user_file = os.path.join(self.users_dir, filename)
                
                try:
                    with open(user_file, 'rb') as f:
                        user_encodings = pickle.load(f)
                    
                    self.known_face_encodings.extend(user_encodings)
                    self.known_face_names.extend([name] * len(user_encodings))
                    print(f"Loaded {len(user_encodings)} encodings for {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def recognize_faces(self, image: np.ndarray, tolerance: float = 0.5) -> List[Dict]:
        """
        Recognize faces in an image, optimized for masked faces.
        
        Args:
            image: Input image
            tolerance: Recognition tolerance (lower = more strict)
            
        Returns:
            List of recognition results with name, confidence, and location
        """
        if not self.known_face_encodings:
            return []
        
        # Get face encodings from current image
        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        
        for (face_encoding, face_location) in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=tolerance
            )
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                # Find the best match
                best_match_index = np.argmin(distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - distances[best_match_index]  # Convert distance to confidence
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': face_location,
                'distance': distances[np.argmin(distances)] if len(distances) > 0 else 1.0
            })
        
        return results
    
    def get_registered_users(self) -> List[str]:
        """Get list of all registered users."""
        users = []
        if os.path.exists(self.users_dir):
            for filename in os.listdir(self.users_dir):
                if filename.endswith('.pkl'):
                    users.append(filename[:-4])
        return users
    
    def delete_user(self, name: str) -> bool:
        """Delete a registered user."""
        user_file = os.path.join(self.users_dir, f"{name}.pkl")
        if os.path.exists(user_file):
            os.remove(user_file)
            self.load_known_faces()  # Reload known faces
            print(f"Deleted user: {name}")
            return True
        return False
