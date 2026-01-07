"""
Vehicle Plate Detection & Registration System - Web Application
Flask-based web interface
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import base64
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vehicle_service import VehicleService
from models.database import Database
import config

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Initialize vehicle service
print("Initializing Vehicle Detection System...")
db = Database(config.DATABASE_PATH)
vehicle_service = VehicleService(database=db, detector_model=config.YOLO_MODEL_PATH, use_gpu=config.USE_GPU)
print("System initialized successfully!")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 for web display."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ==================== WEB ROUTES ====================

@app.route('/')
def index():
    """Home page - Detection Dashboard."""
    stats = vehicle_service.get_stats()
    return render_template('dashboard.html', stats=stats)


@app.route('/registry')
def registry():
    """Vehicle Registry page."""
    vehicles = vehicle_service.get_all_vehicles()
    return render_template('registry.html', vehicles=vehicles)


@app.route('/detections')
def detections_page():
    """Detection history page."""
    recent = vehicle_service.get_recent_detections(limit=100)
    return render_template('detections.html', detections=recent)


# ==================== API ROUTES ====================

@app.route('/api/detect', methods=['POST'])
def detect_plate():
    """API endpoint to detect license plate in uploaded image."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, PNG, or BMP'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        results = vehicle_service.process_image(filepath, log_detection=True)

        if results['success']:
            # Draw detections on image
            image = cv2.imread(filepath)

            for det in results['detections']:
                x1, y1, x2, y2 = det['bbox']
                registered = det['registered']
                plate_num = det['plate_number']

                # Color: green if registered, red if not
                color = (0, 255, 0) if registered else (0, 0, 255)

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

                # Add label
                label = f"{plate_num} ({'REG' if registered else 'NEW'})"
                cv2.putText(image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Save annotated image
            annotated_filename = f"annotated_{filename}"
            annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            cv2.imwrite(annotated_filepath, image)

            # Add image URL to response
            results['image_url'] = f'/uploads/{annotated_filename}'
            results['original_url'] = f'/uploads/{filename}'

        return jsonify(results)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    """Get all registered vehicles."""
    try:
        vehicles = vehicle_service.get_all_vehicles()
        vehicles_data = [v.to_dict() for v in vehicles]
        return jsonify({'success': True, 'vehicles': vehicles_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/vehicles/<plate_number>', methods=['GET'])
def get_vehicle(plate_number):
    """Get specific vehicle by plate number."""
    try:
        vehicle = vehicle_service.get_vehicle_info(plate_number)
        if vehicle:
            return jsonify({'success': True, 'vehicle': vehicle.to_dict()})
        else:
            return jsonify({'success': False, 'error': 'Vehicle not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/vehicles', methods=['POST'])
def register_vehicle():
    """Register a new vehicle."""
    try:
        data = request.get_json()

        plate_number = data.get('plate_number', '').strip()
        owner_name = data.get('owner_name', '').strip()
        vehicle_type = data.get('vehicle_type', '').strip() or None
        color = data.get('color', '').strip() or None
        model = data.get('model', '').strip() or None

        if not plate_number or not owner_name:
            return jsonify({'success': False, 'error': 'Plate number and owner name are required'}), 400

        success = vehicle_service.register_vehicle(
            plate_number=plate_number,
            owner_name=owner_name,
            vehicle_type=vehicle_type,
            color=color,
            model=model
        )

        if success:
            return jsonify({'success': True, 'message': f'Vehicle {plate_number} registered successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to register vehicle (may already exist)'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/vehicles/<plate_number>', methods=['PUT'])
def update_vehicle(plate_number):
    """Update vehicle information."""
    try:
        data = request.get_json()

        success = vehicle_service.update_vehicle_info(plate_number, **data)

        if success:
            return jsonify({'success': True, 'message': 'Vehicle updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Vehicle not found or update failed'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/vehicles/<plate_number>', methods=['DELETE'])
def delete_vehicle(plate_number):
    """Delete a vehicle."""
    try:
        success = vehicle_service.delete_vehicle(plate_number)

        if success:
            return jsonify({'success': True, 'message': 'Vehicle deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Vehicle not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        stats = vehicle_service.get_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent detection logs."""
    try:
        limit = request.args.get('limit', 50, type=int)
        detections = vehicle_service.get_recent_detections(limit=limit)

        detections_data = []
        for det in detections:
            detections_data.append({
                'plate_number': det[0],
                'confidence': det[1],
                'detected_at': det[2],
                'image_path': det[3],
                'status': det[4]
            })

        return jsonify({'success': True, 'detections': detections_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detections/clear', methods=['POST'])
def clear_history():
    """Clear all detection history."""
    try:
        if db.clear_detections():
            return jsonify({'success': True, 'message': 'History cleared successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to clear history'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== STATIC FILES ====================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """404 error handler."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler."""
    return render_template('500.html'), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Vehicle Plate Detection System - Web Interface")
    print("=" * 60)
    print("\nServer starting at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
