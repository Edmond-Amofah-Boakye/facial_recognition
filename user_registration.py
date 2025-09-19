import cv2
import numpy as np
import os
from face_recognition_system import MaskedFaceRecognitionSystem
from typing import List

class UserRegistration:
    def __init__(self):
        """Initialize the user registration system."""
        self.face_system = MaskedFaceRecognitionSystem()
        self.cap = None
    
    def capture_images_from_webcam(self, name: str, num_images: int = 10) -> bool:
        """
        Capture multiple images of a user from webcam for registration.
        
        Args:
            name: User's name
            num_images: Number of images to capture
            
        Returns:
            True if registration successful, False otherwise
        """
        print(f"\n=== Registering User: {name} ===")
        print(f"We'll capture {num_images} images for better recognition.")
        print("Instructions:")
        print("- Look directly at the camera")
        print("- Try different angles (slightly left, right, up, down)")
        print("- Take some photos with a mask and some without")
        print("- Press SPACE to capture an image")
        print("- Press 'q' to quit early")
        print("- Press 'r' to restart capture")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        captured_images = []
        capture_count = 0
        
        while capture_count < num_images:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw instructions on frame
            cv2.putText(frame, f"Capturing for: {name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Images captured: {capture_count}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Quit | R: Restart", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw face detection rectangle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_system.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            cv2.imshow('User Registration - Face Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                if len(faces) > 0:
                    captured_images.append(frame.copy())
                    capture_count += 1
                    print(f"✓ Captured image {capture_count}/{num_images}")
                else:
                    print("⚠ No face detected. Please position your face in the camera.")
            
            elif key == ord('q'):  # Quit
                print("Registration cancelled by user.")
                break
            
            elif key == ord('r'):  # Restart
                captured_images = []
                capture_count = 0
                print("Restarting capture...")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        if len(captured_images) == 0:
            print("No images captured. Registration failed.")
            return False
        
        # Register the user with captured images
        print(f"\nProcessing {len(captured_images)} images...")
        success = self.face_system.register_user(name, captured_images)
        
        if success:
            print(f"✓ Successfully registered {name}!")
            print(f"  - Captured {len(captured_images)} images")
            print(f"  - User data saved to users/{name}.pkl")
        else:
            print(f"✗ Failed to register {name}")
        
        return success
    
    def register_from_images(self, name: str, image_paths: List[str]) -> bool:
        """
        Register a user from existing image files.
        
        Args:
            name: User's name
            image_paths: List of paths to image files
            
        Returns:
            True if registration successful, False otherwise
        """
        images = []
        
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image not found: {path}")
                continue
            
            try:
                image = cv2.imread(path)
                if image is not None:
                    images.append(image)
                    print(f"Loaded: {path}")
                else:
                    print(f"Warning: Could not load image: {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        if not images:
            print("No valid images found for registration.")
            return False
        
        print(f"Registering {name} with {len(images)} images...")
        return self.face_system.register_user(name, images)
    
    def list_registered_users(self):
        """Display all registered users."""
        users = self.face_system.get_registered_users()
        
        if not users:
            print("No users registered yet.")
        else:
            print(f"\nRegistered Users ({len(users)}):")
            for i, user in enumerate(users, 1):
                print(f"  {i}. {user}")
    
    def delete_user(self, name: str) -> bool:
        """Delete a registered user."""
        return self.face_system.delete_user(name)

def main():
    """Main registration interface."""
    registration = UserRegistration()
    
    while True:
        print("\n" + "="*50)
        print("MASKED FACE RECOGNITION - USER REGISTRATION")
        print("="*50)
        print("1. Register new user (webcam)")
        print("2. Register from image files")
        print("3. List registered users")
        print("4. Delete user")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            name = input("Enter user name: ").strip()
            if name:
                num_images = input("Number of images to capture (default 10): ").strip()
                try:
                    num_images = int(num_images) if num_images else 10
                    num_images = max(1, min(num_images, 20))  # Limit between 1-20
                except ValueError:
                    num_images = 10
                
                registration.capture_images_from_webcam(name, num_images)
            else:
                print("Please enter a valid name.")
        
        elif choice == '2':
            name = input("Enter user name: ").strip()
            if name:
                print("Enter image file paths (one per line, empty line to finish):")
                image_paths = []
                while True:
                    path = input("Image path: ").strip()
                    if not path:
                        break
                    image_paths.append(path)
                
                if image_paths:
                    registration.register_from_images(name, image_paths)
                else:
                    print("No image paths provided.")
            else:
                print("Please enter a valid name.")
        
        elif choice == '3':
            registration.list_registered_users()
        
        elif choice == '4':
            registration.list_registered_users()
            name = input("\nEnter name to delete: ").strip()
            if name:
                if registration.delete_user(name):
                    print(f"✓ User '{name}' deleted successfully.")
                else:
                    print(f"✗ User '{name}' not found.")
            else:
                print("Please enter a valid name.")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
