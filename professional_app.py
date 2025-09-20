from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, flash
import cv2
import json
import threading
import time
from professional_face_system import ProfessionalFaceRecognitionSystem
import base64
import numpy as np
import os
from datetime import datetime, timedelta
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate secure secret key

# Initialize the professional face recognition system
face_system = ProfessionalFaceRecognitionSystem()

# Global variables for camera management
camera = None
camera_active = False
current_frame = None
frame_lock = threading.Lock()

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()
        
    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
            
        # Flip image horizontally for mirror effect
        image = cv2.flip(image, 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Store current frame for processing
        with frame_lock:
            global current_frame
            current_frame = image.copy()
        
        # Encode image to JPEG
        ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes()

def generate_frames():
    global camera, camera_active
    while camera_active:
        if camera is None:
            try:
                camera = VideoCamera()
            except Exception as e:
                print(f"Camera initialization failed: {e}")
                time.sleep(1)
                continue
            
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Main dashboard page."""
    stats = face_system.get_system_stats()
    recent_logs = face_system.get_access_logs(limit=10)
    return render_template('professional_index.html', stats=stats, recent_logs=recent_logs)

@app.route('/register')
def register_page():
    """User registration page."""
    return render_template('register.html')

@app.route('/authenticate')
def authenticate_page():
    """Authentication page."""
    return render_template('authenticate.html')

@app.route('/users')
def users_page():
    """User management page."""
    users = face_system.get_all_users()
    return render_template('users.html', users=users)

@app.route('/logs')
def logs_page():
    """Access logs page."""
    logs = face_system.get_access_logs(limit=50)
    return render_template('logs.html', logs=logs)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera feed."""
    global camera_active, camera
    try:
        if not camera_active:
            camera_active = True
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({'success': True, 'message': 'Camera already running'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to start camera: {str(e)}'})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera feed."""
    global camera_active, camera
    try:
        camera_active = False
        if camera:
            del camera
            camera = None
        return jsonify({'success': True, 'message': 'Camera stopped successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to stop camera: {str(e)}'})

@app.route('/api/camera/capture', methods=['POST'])
def capture_frame():
    """Capture current frame."""
    global current_frame
    try:
        with frame_lock:
            if current_frame is not None:
                # Encode frame to base64
                ret, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({'success': True, 'frame': frame_base64})
        
        return jsonify({'success': False, 'message': 'No frame available'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Capture failed: {str(e)}'})

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['full_name', 'email']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'success': False, 'message': f'{field.replace("_", " ").title()} is required'})
        
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip().lower()
        phone = data.get('phone', '').strip()
        department = data.get('department', '').strip()
        face_images = data.get('face_images', [])
        
        # Validate face images
        if not face_images or len(face_images) < 5:
            return jsonify({
                'success': False, 
                'message': 'Please provide at least 5 clear face images for accurate recognition'
            })
        
        # Convert base64 images to OpenCV format
        processed_images = []
        for img_data in face_images:
            try:
                # Remove data URL prefix if present
                if ',' in img_data:
                    img_data = img_data.split(',')[1]
                
                # Decode base64 image
                img_bytes = base64.b64decode(img_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    processed_images.append(img)
                    
            except Exception as e:
                print(f"Failed to process image: {e}")
                continue
        
        if len(processed_images) < 3:
            return jsonify({
                'success': False, 
                'message': 'Failed to process face images. Please ensure images are clear and contain visible faces.'
            })
        
        # Register user
        result = face_system.register_user(
            full_name=full_name,
            email=email,
            phone=phone if phone else None,
            department=department if department else None,
            face_images=processed_images
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """Authenticate user using face recognition."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})
        
        # Authenticate user
        result = face_system.authenticate_user(image)
        
        # Add session management for successful authentication
        if result['success'] and result['user']:
            session['user_id'] = result['user']['user_id']
            session['user_name'] = result['user']['full_name']
            session['login_time'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Authentication error: {str(e)}'})

@app.route('/api/users')
def get_users():
    """Get all registered users."""
    try:
        users = face_system.get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to get users: {str(e)}'})

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user information."""
    try:
        user = face_system.get_user_info(user_id)
        if user:
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to get user: {str(e)}'})

@app.route('/api/users/<user_id>/status', methods=['PUT'])
def update_user_status(user_id):
    """Update user status."""
    try:
        data = request.get_json()
        status = data.get('status')
        
        if status not in ['active', 'inactive', 'suspended']:
            return jsonify({'success': False, 'message': 'Invalid status'})
        
        success = face_system.update_user_status(user_id, status)
        
        if success:
            return jsonify({'success': True, 'message': f'User status updated to {status}'})
        else:
            return jsonify({'success': False, 'message': 'Failed to update user status'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Update failed: {str(e)}'})

@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete user."""
    try:
        success = face_system.delete_user(user_id)
        
        if success:
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to delete user'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Delete failed: {str(e)}'})

@app.route('/api/logs')
def get_access_logs():
    """Get access logs."""
    try:
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 50))
        
        logs = face_system.get_access_logs(user_id=user_id, limit=limit)
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to get logs: {str(e)}'})

@app.route('/api/stats')
def get_system_stats():
    """Get system statistics."""
    try:
        stats = face_system.get_system_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to get stats: {str(e)}'})

@app.route('/api/logs/<int:log_id>', methods=['DELETE'])
def delete_log(log_id):
    """Delete a specific access log entry."""
    try:
        success = face_system.delete_log(log_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Log deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Log not found or deletion failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Delete failed: {str(e)}'})

@app.route('/api/logs/user/<user_id>', methods=['DELETE'])
def delete_user_logs(user_id):
    """Delete all logs for a specific user."""
    try:
        success = face_system.delete_user_logs(user_id)
        
        if success:
            return jsonify({'success': True, 'message': f'All logs for user {user_id} deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'No logs found for this user'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Delete failed: {str(e)}'})

@app.route('/api/logs/clear', methods=['DELETE'])
def clear_all_logs():
    """Clear all access logs."""
    try:
        success = face_system.clear_all_logs()
        
        if success:
            return jsonify({'success': True, 'message': 'All logs cleared successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to clear logs'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Clear failed: {str(e)}'})

@app.route('/api/logs/cleanup', methods=['DELETE'])
def cleanup_old_logs():
    """Delete old access logs."""
    try:
        days = int(request.args.get('days', 30))
        success = face_system.delete_old_logs(days)
        
        if success:
            return jsonify({'success': True, 'message': f'Logs older than {days} days deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to delete old logs'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Cleanup failed: {str(e)}'})

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user."""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    import os
    
    print("🏢 Professional Face Recognition System")
    print("=" * 60)
    print("🚀 Starting professional Flask server...")
    
    # Get port from environment variable (for hosting platforms) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    if port == 5000:
        print("🌐 Open your browser and go to: http://localhost:5000")
    else:
        print(f"🌐 Running on port: {port}")
    
    print("✨ Features:")
    print("   • Professional user registration and management")
    print("   • Secure face-based authentication")
    print("   • Comprehensive access logging and monitoring")
    print("   • Real-time camera integration")
    print("   • Works with face masks and partial occlusion")
    print("   • SQLite database with proper user management")
    print("   • Modern responsive web interface")
    print("=" * 60)
    
    # Use debug=False for production deployment
    debug_mode = port == 5000  # Only debug locally
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
