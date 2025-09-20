import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
from deepface import DeepFace
import base64
from PIL import Image
import io

class MaskedFaceRecognitionSystem:
    def __init__(self, users_dir: str = "users"):
        """
        Initialize the masked face recognition system using DeepFace.
        
        Args:
            users_dir: Directory to store user face encodings
        """
        self.users_dir = users_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create users directory if it doesn't exist
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # Load existing user data
        self.load_known_faces()
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features using DeepFace.
        
        Args:
            image: Input image
            
        Returns:
            Face embedding or None if no face found
        """
        try:
            # DeepFace expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Get face embedding using DeepFace
            embedding = DeepFace.represent(rgb_image, model_name='Facenet', enforce_detection=False)
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return np.array(embedding['embedding'])
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def register_user(self, name: str, images: List) -> bool:
        """
        Register a new user with multiple images.
        
        Args:
            name: User's name
            images: List of base64 encoded images or numpy arrays
            
        Returns:
            True if registration successful, False otherwise
        """
        if not name or not images:
            return False
        
        user_embeddings = []
        
        for img_data in images:
            try:
                # Handle both base64 strings and numpy arrays
                if isinstance(img_data, str):
                    # Decode base64 image
                    img_bytes = base64.b64decode(img_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    # Already a numpy array
                    image = img_data
                
                if image is None:
                    continue
                
                # Extract face embedding
                embedding = self.extract_face_features(image)
                if embedding is not None:
                    user_embeddings.append(embedding)
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if len(user_embeddings) > 0:
            # Store average embedding for the user
            avg_embedding = np.mean(user_embeddings, axis=0)
            
            # Save user data
            user_file = os.path.join(self.users_dir, f"{name}.pkl")
            user_data = {
                'embedding': avg_embedding,
                'sample_count': len(user_embeddings),
                'name': name
            }
            
            with open(user_file, 'wb') as f:
                pickle.dump(user_data, f)
            
            print(f"Successfully registered {name} with {len(user_embeddings)} face embeddings")
            return True
        
        print(f"No faces found in images for {name}")
        return False
    
    def load_known_faces(self):
        """Load all known faces from the users directory."""
        self.known_users = {}
        
        if not os.path.exists(self.users_dir):
            return
        
        for filename in os.listdir(self.users_dir):
            if filename.endswith('.pkl'):
                name = filename[:-4]  # Remove .pkl extension
                user_file = os.path.join(self.users_dir, filename)
                
                try:
                    with open(user_file, 'rb') as f:
                        user_data = pickle.load(f)
                    
                    self.known_users[name] = user_data
                    print(f"Loaded user: {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2
        except:
            return 0.0
    
    def recognize_faces(self, image: np.ndarray, tolerance: float = 0.6) -> List[Dict]:
        """
        Recognize faces in an image using DeepFace.
        
        Args:
            image: Input image
            tolerance: Recognition tolerance (higher = more strict)
            
        Returns:
            List of recognition results with name, confidence, and location
        """
        results = []
        
        try:
            # Detect faces using OpenCV for bounding boxes
            faces = self.face_cascade.detectMultiScale(image, 1.1, 4)
            
            if len(faces) == 0:
                return results
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = image[y:y+h, x:x+w]
                
                if len(self.known_users) == 0:
                    # No users registered
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y, x+w, y+h, x),  # top, right, bottom, left
                        'distance': 1.0
                    })
                    continue
                
                # Extract embedding from face
                face_embedding = self.extract_face_features(face_img)
                
                if face_embedding is None:
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y, x+w, y+h, x),
                        'distance': 1.0
                    })
                    continue
                
                # Compare with known users
                best_match = None
                best_similarity = 0.0
                
                for user_name, user_data in self.known_users.items():
                    stored_embedding = user_data['embedding']
                    similarity = self.calculate_similarity(face_embedding, stored_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = user_name
                
                # Determine if it's a match based on tolerance
                if best_similarity > tolerance:
                    name = best_match
                    confidence = best_similarity
                else:
                    name = "Unknown"
                    confidence = best_similarity
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (y, x+w, y+h, x),  # top, right, bottom, left
                    'distance': 1.0 - best_similarity
                })
                
        except Exception as e:
            print(f"Recognition error: {e}")
            # Return unknown result if error occurs
            results.append({
                'name': 'Unknown',
                'confidence': 0.0,
                'location': (0, 100, 100, 0),
                'distance': 1.0
            })
        
        return results
    
    def get_registered_users(self) -> List[str]:
        """Get list of all registered users."""
        return list(self.known_users.keys()) if hasattr(self, 'known_users') else []
    
    def delete_user(self, name: str) -> bool:
        """Delete a registered user."""
        user_file = os.path.join(self.users_dir, f"{name}.pkl")
        if os.path.exists(user_file):
            os.remove(user_file)
            self.load_known_faces()  # Reload known faces
            print(f"Deleted user: {name}")
            return True
        return False
