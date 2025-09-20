import cv2
import numpy as np
import os
import pickle
import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import base64
from PIL import Image
import io
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalFaceRecognitionSystem:
    def __init__(self, db_path: str = "face_recognition.db", models_dir: str = "models"):
        """
        Professional Face Recognition System with user management and authentication.
        
        Args:
            db_path: Path to SQLite database
            models_dir: Directory to store face models
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize face recognition models
        self._initialize_models()
        
        # Initialize database
        self._initialize_database()
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Professional Face Recognition System initialized on {self.device}")
    
    def _initialize_models(self):
        """Initialize MTCNN and FaceNet models optimized for masked faces."""
        try:
            # More sensitive MTCNN for better face detection with masks
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,  # Increased margin for better face capture
                min_face_size=40,  # Larger minimum face size for better quality
                thresholds=[0.5, 0.6, 0.6],  # Lower thresholds for better detection
                factor=0.8,  # Better scaling factor
                post_process=True,
                device=self.device,
                keep_all=False,
                selection_method='largest'  # Select largest face
            )
            
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info("Face recognition models loaded successfully with mask optimization")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize SQLite database with proper schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                department TEXT,
                role TEXT DEFAULT 'user',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP,
                face_model_path TEXT
            )
        ''')
        
        # Access logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                access_type TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                location TEXT,
                device_info TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # System settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default settings
        default_settings = [
            ('recognition_threshold', '0.6'),
            ('max_failed_attempts', '3'),
            ('session_timeout', '3600'),
            ('require_registration_approval', 'false'),
            ('enable_access_logging', 'true')
        ]
        
        for key, value in default_settings:
            cursor.execute('''
                INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)
            ''', (key, value))
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def register_user(self, full_name: str, email: str, phone: str = None, 
                     department: str = None, face_images: List = None) -> Dict:
        """
        Register a new user in the system.
        
        Args:
            full_name: User's full name
            email: User's email address
            phone: User's phone number (optional)
            department: User's department (optional)
            face_images: List of face images for training
            
        Returns:
            Dictionary with registration result
        """
        try:
            # Generate unique user ID
            user_id = str(uuid.uuid4())[:8].upper()
            
            # Validate email format
            if '@' not in email or '.' not in email:
                return {'success': False, 'message': 'Invalid email format'}
            
            # Check if email already exists
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT email FROM users WHERE email = ?', (email,))
            if cursor.fetchone():
                conn.close()
                return {'success': False, 'message': 'Email already registered'}
            
            # Process face images if provided
            face_model_path = None
            if face_images and len(face_images) >= 5:
                face_embeddings = []
                
                for img_data in face_images:
                    try:
                        # Convert image data
                        if isinstance(img_data, str):
                            img_bytes = base64.b64decode(img_data)
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        else:
                            image = img_data
                        
                        # Extract face embedding
                        embedding = self._extract_face_embedding(image)
                        if embedding is not None:
                            face_embeddings.append(embedding)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process face image: {e}")
                        continue
                
                if len(face_embeddings) >= 3:
                    # Save face model
                    avg_embedding = np.mean(face_embeddings, axis=0)
                    face_model_path = os.path.join(self.models_dir, f"{user_id}_face_model.pkl")
                    
                    face_model = {
                        'user_id': user_id,
                        'embedding': avg_embedding,
                        'sample_count': len(face_embeddings),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    with open(face_model_path, 'wb') as f:
                        pickle.dump(face_model, f)
                else:
                    return {'success': False, 'message': 'Insufficient quality face images. Please provide at least 3 clear face images.'}
            
            # Insert user into database
            cursor.execute('''
                INSERT INTO users (user_id, full_name, email, phone, department, face_model_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, full_name, email, phone, department, face_model_path))
            
            conn.commit()
            conn.close()
            
            # Log registration
            self._log_access(user_id, 'registration', 1.0)
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id,
                'has_face_model': face_model_path is not None
            }
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def authenticate_user(self, image: np.ndarray, threshold: float = None, use_smoothing: bool = True) -> Dict:
        """
        Authenticate user using face recognition with improved stability.
        
        Args:
            image: Input image for recognition
            threshold: Recognition threshold (optional)
            use_smoothing: Whether to use confidence smoothing (optional)
            
        Returns:
            Dictionary with authentication result
        """
        try:
            if threshold is None:
                threshold = float(self._get_setting('recognition_threshold', 0.7))  # Increased threshold
            
            # Extract face embedding from input image - single attempt
            face_embedding = self._extract_face_embedding(image)
            
            if face_embedding is None:
                return {
                    'success': False,
                    'message': 'No face detected in image',
                    'user': None,
                    'confidence': 0.0
                }
            
            # Load all user face models
            best_distances = []
            best_user = None
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, full_name, email, department, role, face_model_path, status
                FROM users WHERE face_model_path IS NOT NULL AND status = 'active'
            ''')
            
            users = cursor.fetchall()
            conn.close()
            
            for user_data in users:
                user_id, full_name, email, department, role, face_model_path, status = user_data
                
                if not os.path.exists(face_model_path):
                    continue
                
                try:
                    with open(face_model_path, 'rb') as f:
                        face_model = pickle.load(f)
                    
                    stored_embedding = face_model['embedding']
                    
                    # Calculate multiple distance metrics for better accuracy
                    euclidean_dist = np.linalg.norm(face_embedding - stored_embedding)
                    cosine_dist = 1 - np.dot(face_embedding, stored_embedding) / (
                        np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    # Combine distances for more stable recognition
                    combined_distance = (euclidean_dist * 0.7) + (cosine_dist * 0.3)
                    
                    best_distances.append({
                        'distance': combined_distance,
                        'user': {
                            'user_id': user_id,
                            'full_name': full_name,
                            'email': email,
                            'department': department,
                            'role': role
                        }
                    })
                        
                except Exception as e:
                    logger.warning(f"Failed to load face model for user {user_id}: {e}")
                    continue
            
            if not best_distances:
                return {
                    'success': False,
                    'message': 'No registered users found',
                    'user': None,
                    'confidence': 0.0
                }
            
            # Sort by distance and get the best match
            best_distances.sort(key=lambda x: x['distance'])
            best_match = best_distances[0]
            best_distance = best_match['distance']
            best_user = best_match['user']
            
            # Enhanced recognition logic for masked faces
            # Use adaptive threshold based on the best match quality
            adaptive_threshold = min(threshold * 1.2, 0.85)  # Allow slightly higher threshold for masks
            
            if best_distance < adaptive_threshold:
                # Calculate confidence with mask-aware scoring
                raw_confidence = max(0.0, 1.0 - (best_distance / adaptive_threshold))
                
                # Boost confidence for consistent recognition (mask tolerance)
                if raw_confidence > 0.5:
                    # Apply mask-aware confidence boost
                    mask_boost = min(0.15, (0.7 - best_distance) * 0.3) if best_distance < 0.7 else 0
                    raw_confidence = min(0.99, raw_confidence + mask_boost)
                
                # Apply confidence smoothing for more stable results
                if use_smoothing:
                    confidence = self._smooth_confidence(raw_confidence, best_user['user_id'])
                else:
                    confidence = raw_confidence
                
                # Ensure minimum confidence for successful recognition
                confidence = float(min(0.99, max(0.65, confidence)))  # Clamp between 65-99%
                
                # Additional validation: check if this is significantly better than second best
                if len(best_distances) > 1:
                    second_best_distance = best_distances[1]['distance']
                    distance_gap = second_best_distance - best_distance
                    
                    # Require significant gap for high confidence recognition
                    if distance_gap < 0.1 and confidence > 0.8:
                        confidence = min(confidence, 0.75)  # Reduce confidence if matches are too close
                
                # Update last seen
                self._update_last_seen(best_user['user_id'])
                
                # Log successful access
                self._log_access(best_user['user_id'], 'authentication_success', confidence)
                
                return {
                    'success': True,
                    'message': f'Welcome, {best_user["full_name"]}! (Mask-tolerant recognition)',
                    'user': best_user,
                    'confidence': confidence
                }
            else:
                # Check if it's a borderline case that might be a masked face
                if best_distance < threshold * 1.5:  # Within extended range
                    # Log as potential masked face attempt
                    self._log_access(best_user['user_id'], 'authentication_failed_possible_mask', 
                                   max(0.0, 1.0 - (best_distance / (threshold * 1.5))))
                    
                    return {
                        'success': False,
                        'message': f'Face partially recognized but confidence too low. If wearing a mask, please ensure good lighting and face the camera directly.',
                        'user': None,
                        'confidence': max(0.0, 1.0 - (best_distance / threshold))
                    }
                else:
                    # Log failed access attempt
                    self._log_access('unknown', 'authentication_failed', 0.0)
                    
                    return {
                        'success': False,
                        'message': 'Face not recognized. Please try again or register.',
                        'user': None,
                        'confidence': max(0.0, 1.0 - (best_distance / threshold)) if best_distance != float('inf') else 0.0
                    }
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {
                'success': False,
                'message': 'Authentication system error',
                'user': None,
                'confidence': 0.0
            }
    
    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image - simplified approach."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Simple face detection and extraction
            face_tensor = self.mtcnn(pil_image)
            
            if face_tensor is not None:
                # Get embedding using FaceNet
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.facenet(face_tensor)
                
                return embedding.cpu().numpy().flatten()
            
            return None
            
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {e}")
            return None
    
    def _preprocess_for_face_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocess image in multiple ways to improve face detection with masks."""
        processed_images = []
        
        try:
            # Original image
            processed_images.append(image.copy())
            
            # Histogram equalization for better contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            processed_images.append(equalized_bgr)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_gray = clahe.apply(gray)
            clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
            processed_images.append(clahe_bgr)
            
            # Gamma correction for better lighting
            gamma = 1.2
            gamma_corrected = np.power(image / 255.0, gamma)
            gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
            processed_images.append(gamma_corrected)
            
            # Slight blur to reduce noise
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            processed_images.append(blurred)
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return [image]  # Return original image if preprocessing fails
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information by user ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, full_name, email, phone, department, role, status, 
                       created_at, last_seen, face_model_path
                FROM users WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'user_id': result[0],
                    'full_name': result[1],
                    'email': result[2],
                    'phone': result[3],
                    'department': result[4],
                    'role': result[5],
                    'status': result[6],
                    'created_at': result[7],
                    'last_seen': result[8],
                    'has_face_model': result[9] is not None
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all registered users."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, full_name, email, department, role, status, 
                       created_at, last_seen, face_model_path
                FROM users ORDER BY created_at DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            users = []
            for result in results:
                users.append({
                    'user_id': result[0],
                    'full_name': result[1],
                    'email': result[2],
                    'department': result[3],
                    'role': result[4],
                    'status': result[5],
                    'created_at': result[6],
                    'last_seen': result[7],
                    'has_face_model': result[8] is not None
                })
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            return []
    
    def get_access_logs(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """Get access logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT al.user_id, u.full_name, al.access_type, al.confidence, 
                           al.timestamp, al.location, al.device_info
                    FROM access_logs al
                    LEFT JOIN users u ON al.user_id = u.user_id
                    WHERE al.user_id = ?
                    ORDER BY al.timestamp DESC LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT al.user_id, u.full_name, al.access_type, al.confidence, 
                           al.timestamp, al.location, al.device_info
                    FROM access_logs al
                    LEFT JOIN users u ON al.user_id = u.user_id
                    ORDER BY al.timestamp DESC LIMIT ?
                ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            logs = []
            for result in results:
                logs.append({
                    'user_id': result[0],
                    'full_name': result[1] or 'Unknown',
                    'access_type': result[2],
                    'confidence': result[3],
                    'timestamp': result[4],
                    'location': result[5],
                    'device_info': result[6]
                })
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to get access logs: {e}")
            return []
    
    def update_user_status(self, user_id: str, status: str) -> bool:
        """Update user status (active/inactive/suspended)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (status, user_id))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                self._log_access(user_id, f'status_changed_to_{status}', 1.0)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update user status: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and associated data."""
        try:
            # Get user info first
            user_info = self.get_user_info(user_id)
            if not user_info:
                return False
            
            # Delete face model file
            if user_info['has_face_model']:
                face_model_path = os.path.join(self.models_dir, f"{user_id}_face_model.pkl")
                if os.path.exists(face_model_path):
                    os.remove(face_model_path)
            
            # Delete from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                logger.info(f"User {user_id} deleted successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False
    
    def delete_log(self, log_id: int) -> bool:
        """Delete a specific access log entry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM access_logs WHERE id = ?', (log_id,))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                logger.info(f"Access log {log_id} deleted successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete log: {e}")
            return False
    
    def delete_user_logs(self, user_id: str) -> bool:
        """Delete all access logs for a specific user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM access_logs WHERE user_id = ?', (user_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} access logs for user {user_id}")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete user logs: {e}")
            return False
    
    def clear_all_logs(self) -> bool:
        """Clear all access logs from the system."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM access_logs')
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleared all access logs ({deleted_count} entries)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear logs: {e}")
            return False
    
    def delete_old_logs(self, days: int = 30) -> bool:
        """Delete access logs older than specified days."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM access_logs 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old access logs (older than {days} days)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete old logs: {e}")
            return False
    
    def _update_last_seen(self, user_id: str):
        """Update user's last seen timestamp."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update last seen: {e}")
    
    def _log_access(self, user_id: str, access_type: str, confidence: float, 
                   location: str = None, device_info: str = None):
        """Log access attempt."""
        try:
            if not self._get_setting('enable_access_logging', 'true') == 'true':
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO access_logs (user_id, access_type, confidence, location, device_info)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, access_type, confidence, location, device_info))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log access: {e}")
    
    def _smooth_confidence(self, raw_confidence: float, user_id: str) -> float:
        """Apply confidence smoothing to reduce fluctuations."""
        try:
            # Get recent confidence values for this user
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT confidence FROM access_logs 
                WHERE user_id = ? AND access_type = 'authentication_success' 
                AND timestamp > datetime('now', '-5 minutes')
                ORDER BY timestamp DESC LIMIT 5
            ''', (user_id,))
            
            recent_confidences = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if recent_confidences:
                # Apply exponential moving average for smoothing
                alpha = 0.3  # Smoothing factor
                smoothed = raw_confidence
                
                for prev_conf in recent_confidences:
                    if prev_conf:
                        smoothed = alpha * raw_confidence + (1 - alpha) * prev_conf
                        break
                
                return smoothed
            else:
                return raw_confidence
                
        except Exception as e:
            logger.error(f"Failed to smooth confidence: {e}")
            return raw_confidence
    
    def _get_setting(self, key: str, default_value: str = None) -> str:
        """Get system setting value."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM system_settings WHERE key = ?', (key,))
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else default_value
        except Exception as e:
            logger.error(f"Failed to get setting: {e}")
            return default_value
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total users
            cursor.execute('SELECT COUNT(*) FROM users')
            total_users = cursor.fetchone()[0]
            
            # Active users
            cursor.execute('SELECT COUNT(*) FROM users WHERE status = "active"')
            active_users = cursor.fetchone()[0]
            
            # Users with face models
            cursor.execute('SELECT COUNT(*) FROM users WHERE face_model_path IS NOT NULL')
            users_with_faces = cursor.fetchone()[0]
            
            # Recent access attempts (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM access_logs 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            recent_access = cursor.fetchone()[0]
            
            # Successful authentications (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM access_logs 
                WHERE access_type = 'authentication_success' 
                AND timestamp > datetime('now', '-1 day')
            ''')
            successful_auth = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'users_with_faces': users_with_faces,
                'recent_access_attempts': recent_access,
                'successful_authentications_24h': successful_auth,
                'system_uptime': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
