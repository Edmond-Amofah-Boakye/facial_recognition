from flask import Flask, render_template, Response, request, jsonify
import cv2
import json
import threading
import time
from face_recognition_system import MaskedFaceRecognitionSystem
import base64
import numpy as np

app = Flask(__name__)

# Global variables
face_system = MaskedFaceRecognitionSystem()
camera = None
camera_active = False
recognition_active = False
current_frame = None
frame_lock = threading.Lock()

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
            
        # Flip image horizontally for mirror effect
        image = cv2.flip(image, 1)
        
        # Add status text only (no face detection overlay on video)
        status_text = "Recognition: ON" if recognition_active else "Recognition: OFF"
        status_color = (0, 255, 0) if recognition_active else (0, 0, 255)
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Store current frame for registration and recognition
        with frame_lock:
            global current_frame
            current_frame = image.copy()
        
        # Encode image to JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

def generate_frames():
    global camera
    while camera_active:
        if camera is None:
            camera = VideoCamera()
            
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    users = face_system.get_registered_users()
    return render_template('index.html', users=users)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, camera
    try:
        camera_active = True
        return jsonify({'success': True, 'message': 'Camera started'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active, camera, recognition_active
    camera_active = False
    recognition_active = False
    if camera:
        del camera
        camera = None
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    global recognition_active
    
    users = face_system.get_registered_users()
    if not users and not recognition_active:
        return jsonify({'success': False, 'message': 'Please register at least one user first'})
    
    recognition_active = not recognition_active
    status = 'started' if recognition_active else 'stopped'
    return jsonify({'success': True, 'message': f'Recognition {status}', 'active': recognition_active})

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global current_frame
    with frame_lock:
        if current_frame is not None:
            # Encode frame to base64
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'frame': frame_base64})
    return jsonify({'success': False, 'message': 'No frame available'})

@app.route('/register_user', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        images_data = data.get('images', [])
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})
            
        if len(images_data) < 5:
            return jsonify({'success': False, 'message': 'At least 5 images are required'})
        
        # Convert base64 images to OpenCV format
        images = []
        for img_data in images_data:
            # Decode base64 image
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
        
        if not images:
            return jsonify({'success': False, 'message': 'No valid images received'})
        
        # Register user
        success = face_system.register_user(name, images)
        
        if success:
            return jsonify({'success': True, 'message': f'User {name} registered successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Registration failed. No faces detected in images.'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_users', methods=['GET'])
def get_users():
    users = face_system.get_registered_users()
    return jsonify({'users': users})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        
        if not name:
            return jsonify({'success': False, 'message': 'Name is required'})
        
        success = face_system.delete_user(name)
        
        if success:
            return jsonify({'success': True, 'message': f'User {name} deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'User not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/get_recognition_results', methods=['GET'])
def get_recognition_results():
    global current_frame, recognition_active
    
    try:
        if not recognition_active or current_frame is None:
            return jsonify({'success': False, 'results': []})
        
        with frame_lock:
            frame_copy = current_frame.copy()
        
        # Get real recognition results
        results = face_system.recognize_faces(frame_copy, tolerance=0.5)
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}', 'results': []})

if __name__ == '__main__':
    import os
    
    print("🎭 Masked Face Recognition Web App")
    print("=" * 50)
    print("🌐 Starting Flask server...")
    
    # Get port from environment variable (for hosting platforms) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    if port == 5000:
        print("📱 Open your browser and go to: http://localhost:5000")
    else:
        print(f"🌐 Running on port: {port}")
    
    print("🎯 Features:")
    print("   • Smooth real-time camera feed")
    print("   • Register users with webcam")
    print("   • Real-time face recognition")
    print("   • Works with face masks!")
    print("=" * 50)
    
    # Use debug=False for production deployment
    debug_mode = port == 5000  # Only debug locally
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
