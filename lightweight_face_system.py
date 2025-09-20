import cv2
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
import base64
from PIL import Image
import io
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms

class PowerfulLightweightFaceRecognition:
    def __init__(self, users_dir: str = "users"):
        """
        Initialize lightweight but powerful face recognition using MTCNN + FaceNet.
        This approach is much lighter than TensorFlow but still very accurate.
        
        Args:
            users_dir: Directory to store user face encodings
        """
        self.users_dir = users_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection (lightweight)
        self.mtcnn = MTCNN(
            image_size=160, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # MTCNN thresholds
            factor=0.709, 
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model (pre-trained, lightweight)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Create users directory if it doesn't exist
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # Load existing user data
        self.load_known_faces()
        
        print(f"Initialized on device: {self.device}")
    
    def extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using MTCNN + FaceNet.
        This works well even with masks as it focuses on eye region and overall face structure.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Face embedding or None if no face found
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect and extract face using MTCNN
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is None:
                return None
            
            # Get embedding using FaceNet
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
            
            # Convert to numpy array
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
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
                embedding = self.extract_face_embedding(image)
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
    
    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (lower is more similar)
        """
        try:
            return np.linalg.norm(embedding1 - embedding2)
        except:
            return float('inf')
    
    def recognize_faces(self, image: np.ndarray, tolerance: float = 0.6) -> List[Dict]:
        """
        Recognize faces in an image using MTCNN + FaceNet.
        
        Args:
            image: Input image
            tolerance: Recognition tolerance (lower = more strict)
            
        Returns:
            List of recognition results with name, confidence, and location
        """
        results = []
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Detect faces and get bounding boxes
            boxes, _ = self.mtcnn.detect(pil_image)
            
            if boxes is None:
                return results
            
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Extract face region
                face_img = image[y1:y2, x1:x2]
                
                if len(self.known_users) == 0:
                    # No users registered
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y1, x2, y2, x1),  # top, right, bottom, left
                        'distance': float('inf')
                    })
                    continue
                
                # Extract embedding from face
                face_embedding = self.extract_face_embedding(face_img)
                
                if face_embedding is None:
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y1, x2, y2, x1),
                        'distance': float('inf')
                    })
                    continue
                
                # Compare with known users
                best_match = None
                best_distance = float('inf')
                
                for user_name, user_data in self.known_users.items():
                    stored_embedding = user_data['embedding']
                    distance = self.calculate_distance(face_embedding, stored_embedding)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = user_name
                
                # Determine if it's a match based on tolerance
                if best_distance < tolerance:
                    name = best_match
                    confidence = max(0.0, 1.0 - (best_distance / tolerance))
                else:
                    name = "Unknown"
                    confidence = max(0.0, 1.0 - (best_distance / tolerance))
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (y1, x2, y2, x1),  # top, right, bottom, left
                    'distance': best_distance
                })
                
        except Exception as e:
            print(f"Recognition error: {e}")
            # Return unknown result if error occurs
            results.append({
                'name': 'Unknown',
                'confidence': 0.0,
                'location': (0, 100, 100, 0),
                'distance': float('inf')
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


# Alternative even lighter version using only OpenCV and basic ML
class UltraLightweightFaceRecognition:
    def __init__(self, users_dir: str = "users"):
        """
        Ultra lightweight face recognition using only OpenCV and basic features.
        No external ML libraries required except OpenCV.
        """
        self.users_dir = users_dir
        
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create users directory if it doesn't exist
        if not os.path.exists(self.users_dir):
            os.makedirs(self.users_dir)
        
        # Load existing user data
        self.load_known_faces()
    
    def extract_face_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face features using OpenCV only.
        Focuses on eye region and upper face (works with masks).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Take the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract upper half of face (eye region, forehead)
            upper_face = gray[y:y+h//2, x:x+w]
            
            if upper_face.size == 0:
                return None
            
            # Resize to standard size
            upper_face = cv2.resize(upper_face, (100, 50))
            
            # Extract multiple types of features
            features = []
            
            # 1. Histogram features
            hist = cv2.calcHist([upper_face], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            features.extend(hist[::8])  # Downsample to reduce size
            
            # 2. LBP (Local Binary Pattern) features
            lbp = self.calculate_lbp(upper_face)
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            lbp_hist = lbp_hist.flatten()
            lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist[::8])  # Downsample
            
            # 3. Edge features
            edges = cv2.Canny(upper_face, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            edge_hist = edge_hist.flatten()
            edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
            features.extend(edge_hist[::16])  # Downsample more
            
            # 4. Statistical features
            features.extend([
                np.mean(upper_face),
                np.std(upper_face),
                np.var(upper_face),
                np.median(upper_face)
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """Simple LBP calculation."""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp
    
    def register_user(self, name: str, images: List) -> bool:
        """Register a new user."""
        if not name or not images:
            return False
        
        user_features = []
        
        for img_data in images:
            try:
                if isinstance(img_data, str):
                    img_bytes = base64.b64decode(img_data)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    image = img_data
                
                if image is None:
                    continue
                
                features = self.extract_face_features(image)
                if features is not None:
                    user_features.append(features)
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if len(user_features) > 0:
            avg_features = np.mean(user_features, axis=0)
            
            user_file = os.path.join(self.users_dir, f"{name}.pkl")
            user_data = {
                'features': avg_features,
                'sample_count': len(user_features),
                'name': name
            }
            
            with open(user_file, 'wb') as f:
                pickle.dump(user_data, f)
            
            print(f"Successfully registered {name} with {len(user_features)} samples")
            return True
        
        return False
    
    def load_known_faces(self):
        """Load known faces."""
        self.known_users = {}
        
        if not os.path.exists(self.users_dir):
            return
        
        for filename in os.listdir(self.users_dir):
            if filename.endswith('.pkl'):
                name = filename[:-4]
                user_file = os.path.join(self.users_dir, filename)
                
                try:
                    with open(user_file, 'rb') as f:
                        user_data = pickle.load(f)
                    self.known_users[name] = user_data
                    print(f"Loaded user: {name}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        try:
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return (similarity + 1) / 2  # Convert to 0-1 range
        except:
            return 0.0
    
    def recognize_faces(self, image: np.ndarray, tolerance: float = 0.6) -> List[Dict]:
        """Recognize faces."""
        results = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                
                if len(self.known_users) == 0:
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y, x+w, y+h, x),
                        'distance': 1.0
                    })
                    continue
                
                face_features = self.extract_face_features(face_img)
                
                if face_features is None:
                    results.append({
                        'name': 'Unknown',
                        'confidence': 0.0,
                        'location': (y, x+w, y+h, x),
                        'distance': 1.0
                    })
                    continue
                
                best_match = None
                best_similarity = 0.0
                
                for user_name, user_data in self.known_users.items():
                    stored_features = user_data['features']
                    similarity = self.calculate_similarity(face_features, stored_features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = user_name
                
                if best_similarity > tolerance:
                    name = best_match
                    confidence = best_similarity
                else:
                    name = "Unknown"
                    confidence = best_similarity
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (y, x+w, y+h, x),
                    'distance': 1.0 - best_similarity
                })
        
        except Exception as e:
            print(f"Recognition error: {e}")
            results.append({
                'name': 'Unknown',
                'confidence': 0.0,
                'location': (0, 100, 100, 0),
                'distance': 1.0
            })
        
        return results
    
    def get_registered_users(self) -> List[str]:
        """Get registered users."""
        return list(self.known_users.keys()) if hasattr(self, 'known_users') else []
    
    def delete_user(self, name: str) -> bool:
        """Delete a user."""
        user_file = os.path.join(self.users_dir, f"{name}.pkl")
        if os.path.exists(user_file):
            os.remove(user_file)
            self.load_known_faces()
            print(f"Deleted user: {name}")
            return True
        return False
