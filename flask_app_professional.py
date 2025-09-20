from flask import Flask, render_template, request, jsonify, redirect, url_for
from professional_face_system import ProfessionalFaceRecognitionSystem
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

# Initialize the professional face recognition system
face_system = ProfessionalFaceRecognitionSystem()

@app.route('/')
def dashboard():
    """Main dashboard with system statistics."""
    stats = face_system.get_system_stats()
    return render_template('professional_index.html', stats=stats)

@app.route('/register')
def register():
    """User registration page."""
    return render_template('register.html')

@app.route('/authenticate')
def authenticate():
    """User authentication page."""
    return render_template('authenticate.html')

@app.route('/users')
def users():
    """User management page."""
    users = face_system.get_all_users()
    return render_template('users.html', users=users)

@app.route('/logs')
def logs():
    """Access logs page."""
    logs = face_system.get_access_logs(limit=50)
    return render_template('logs.html', logs=logs)

# API Routes
@app.route('/api/register', methods=['POST'])
def api_register():
    """Register a new user."""
    try:
        data = request.get_json()
        
        full_name = data.get('full_name', '').strip()
        email = data.get('email', '').strip()
        phone = data.get('phone', '').strip()
        department = data.get('department', '').strip()
        face_images = data.get('face_images', [])
        
        if not full_name or not email:
            return jsonify({'success': False, 'message': 'Full name and email are required'})
        
        if len(face_images) < 5:
            return jsonify({'success': False, 'message': 'At least 5 face images are required'})
        
        # Register user
        result = face_system.register_user(
            full_name=full_name,
            email=email,
            phone=phone if phone else None,
            department=department if department else None,
            face_images=face_images
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Registration error: {str(e)}'})

@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    """Authenticate user using face recognition."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'})
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image format'})
        
        # Authenticate user
        result = face_system.authenticate_user(image)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Authentication error: {str(e)}'})

@app.route('/api/users')
def api_users():
    """Get all users."""
    try:
        users = face_system.get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/users/<user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """Delete a user."""
    try:
        success = face_system.delete_user(user_id)
        if success:
