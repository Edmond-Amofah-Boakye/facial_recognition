import cv2
import numpy as np
import time
from face_recognition_system import MaskedFaceRecognitionSystem
from user_registration import UserRegistration

class MaskedFaceRecognitionApp:
    def __init__(self):
        """Initialize the masked face recognition application."""
        self.face_system = MaskedFaceRecognitionSystem()
        self.cap = None
        self.recognition_active = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Recognition settings
        self.tolerance = 0.5  # Lower = more strict recognition
        self.confidence_threshold = 0.4  # Minimum confidence to display name
    
    def draw_face_info(self, frame, results):
        """
        Draw face recognition results on the frame.
        
        Args:
            frame: Video frame
            results: Recognition results from face_system.recognize_faces()
        """
        for result in results:
            top, right, bottom, left = result['location']
            name = result['name']
            confidence = result['confidence']
            
            # Choose color based on recognition status
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                display_name = "Unknown"
            else:
                if confidence >= self.confidence_threshold:
                    color = (0, 255, 0)  # Green for recognized
                    display_name = f"{name} ({confidence:.2f})"
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                    display_name = f"{name}? ({confidence:.2f})"
            
            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name background
            text_width = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            cv2.rectangle(frame, (left, bottom), (left + text_width + 10, bottom + 25), color, -1)
            
            # Draw name text
            cv2.putText(frame, display_name, (left + 5, bottom + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw eye region indicator (for masked face recognition)
            eye_region_bottom = top + int((bottom - top) * 0.6)
            cv2.rectangle(frame, (left, top), (right, eye_region_bottom), (255, 255, 0), 1)
            cv2.putText(frame, "Eye Region", (left, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def draw_ui_elements(self, frame):
        """Draw UI elements on the frame."""
        height, width = frame.shape[:2]
        
        # Draw title
        cv2.putText(frame, "Masked Face Recognition System", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw recognition status
        status = "ACTIVE" if self.recognition_active else "PAUSED"
        status_color = (0, 255, 0) if self.recognition_active else (0, 0, 255)
        cv2.putText(frame, f"Recognition: {status}", (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw tolerance setting
        cv2.putText(frame, f"Tolerance: {self.tolerance:.2f}", (10, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw controls
        controls = [
            "SPACE: Toggle Recognition",
            "R: Register New User",
            "T: Adjust Tolerance",
            "Q: Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (width - 250, height - 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def adjust_tolerance(self):
        """Adjust recognition tolerance."""
        print(f"\nCurrent tolerance: {self.tolerance:.2f}")
        print("Lower values = more strict recognition")
        print("Higher values = more lenient recognition")
        print("Recommended range: 0.3 - 0.7")
        
        try:
            new_tolerance = float(input("Enter new tolerance (0.1 - 1.0): "))
            if 0.1 <= new_tolerance <= 1.0:
                self.tolerance = new_tolerance
                print(f"Tolerance set to {self.tolerance:.2f}")
            else:
                print("Invalid range. Keeping current tolerance.")
        except ValueError:
            print("Invalid input. Keeping current tolerance.")
    
    def register_new_user_quick(self):
        """Quick user registration during live recognition."""
        print("\n" + "="*40)
        print("QUICK USER REGISTRATION")
        print("="*40)
        
        name = input("Enter user name: ").strip()
        if not name:
            print("Invalid name. Registration cancelled.")
            return
        
        print("Position yourself in front of the camera.")
        print("Press ENTER when ready to start capturing...")
        input()
        
        # Temporarily pause recognition
        was_active = self.recognition_active
        self.recognition_active = False
        
        # Use registration system
        registration = UserRegistration()
        success = registration.capture_images_from_webcam(name, 8)  # Quick capture
        
        if success:
            # Reload the face system to include new user
            self.face_system.load_known_faces()
            print(f"✓ {name} registered and ready for recognition!")
        
        # Resume recognition
        self.recognition_active = was_active
    
    def run(self):
        """Run the main recognition application."""
        print("Starting Masked Face Recognition System...")
        print("Make sure you have registered users first!")
        
        # Check if any users are registered
        users = self.face_system.get_registered_users()
        if not users:
            print("\n⚠ WARNING: No users registered!")
            print("Please run 'python user_registration.py' first to register users.")
            print("Or press 'R' during recognition to register users quickly.")
        else:
            print(f"✓ Found {len(users)} registered users: {', '.join(users)}")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*50)
        print("MASKED FACE RECOGNITION - LIVE MODE")
        print("="*50)
        print("Controls:")
        print("  SPACE: Toggle recognition on/off")
        print("  R: Register new user")
        print("  T: Adjust tolerance")
        print("  Q: Quit")
        print("="*50)
        
        self.recognition_active = True
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Perform face recognition if active
                if self.recognition_active:
                    results = self.face_system.recognize_faces(frame, tolerance=self.tolerance)
                    self.draw_face_info(frame, results)
                
                # Draw UI elements
                self.draw_ui_elements(frame)
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Masked Face Recognition System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord(' '):  # Toggle recognition
                    self.recognition_active = not self.recognition_active
                    status = "ACTIVE" if self.recognition_active else "PAUSED"
                    print(f"Recognition {status}")
                elif key == ord('r'):  # Register new user
                    self.register_new_user_quick()
                elif key == ord('t'):  # Adjust tolerance
                    self.adjust_tolerance()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed.")

def main():
    """Main entry point."""
    print("="*60)
    print("MASKED FACE RECOGNITION SYSTEM")
    print("="*60)
    print("This system can recognize faces even when wearing masks!")
    print("It focuses on eye region and upper face features.")
    print("="*60)
    
    app = MaskedFaceRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()
