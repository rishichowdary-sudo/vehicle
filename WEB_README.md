# Vehicle Plate Detection System - Web Application

A modern web-based license plate detection and registration system running on localhost. Access from any browser!

## Features

- **Modern Web Interface**: Clean, responsive design works on desktop and mobile
- **Real-time Detection**: Upload images and get instant plate detection results
- **Vehicle Registry**: Manage registered vehicles with search functionality
- **Detection History**: Track all plate detections with confidence scores
- **RESTful API**: Full API for programmatic access

## Quick Start

### 1. Install Dependencies

```bash
cd "C:\Users\Rishichowdary-3925\Downloads\Vehicle Registration System"
pip install -r requirements.txt
```

**Note**: First installation takes 5-10 minutes (downloads AI models)

### 2. Run the Web Application

```bash
python webapp.py
```

### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

The server will start and you'll see:
```
Vehicle Plate Detection System - Web Interface
============================================================
Server starting at: http://localhost:5000
Press Ctrl+C to stop the server
```

## Using the Web Interface

### Detection Dashboard (Home Page)

1. **Upload an Image**:
   - Click the upload area or "Choose Image" button
   - Or drag and drop an image
   - Supported formats: JPG, PNG, BMP

2. **Detect Plate**:
   - Click "Detect Plate" button
   - Wait for processing (2-5 seconds)
   - See results with annotated image

3. **Quick Registration**:
   - If plate is unregistered, form auto-fills
   - Enter owner details
   - Click "Register Vehicle"

### Vehicle Registry

1. **View All Vehicles**:
   - See table of all registered vehicles
   - Shows plate number, owner, type, color, model, date

2. **Search**:
   - Use search box to filter by plate or owner
   - Results update in real-time

3. **Add Vehicle**:
   - Click "+ Add Vehicle"
   - Fill in the form
   - Click "Save Vehicle"

4. **Delete Vehicle**:
   - Click "Delete" button on any vehicle
   - Confirm deletion

### Detection History

- View all past detections
- See confidence scores
- Check registration status
- Auto-refreshes every 30 seconds

## API Endpoints

The web app provides a REST API for integration:

### Detection

```bash
# Detect plate in image
POST /api/detect
Content-Type: multipart/form-data
Body: { image: <file> }

Response:
{
  "success": true,
  "detections": [
    {
      "plate_number": "ABC1234",
      "detection_confidence": 0.95,
      "ocr_confidence": 0.87,
      "registered": true,
      "vehicle_info": { ... }
    }
  ],
  "image_url": "/uploads/annotated_xxx.jpg"
}
```

### Vehicle Management

```bash
# Get all vehicles
GET /api/vehicles

# Get specific vehicle
GET /api/vehicles/<plate_number>

# Register vehicle
POST /api/vehicles
Content-Type: application/json
Body: {
  "plate_number": "ABC1234",
  "owner_name": "John Doe",
  "vehicle_type": "Car",
  "color": "Blue",
  "model": "Toyota Camry"
}

# Update vehicle
PUT /api/vehicles/<plate_number>
Content-Type: application/json
Body: { ... fields to update ... }

# Delete vehicle
DELETE /api/vehicles/<plate_number>
```

### Statistics

```bash
# Get system stats
GET /api/stats

Response:
{
  "success": true,
  "stats": {
    "total_vehicles": 25,
    "recent_detections": 10
  }
}
```

### Detection History

```bash
# Get recent detections
GET /api/detections?limit=100

Response:
{
  "success": true,
  "detections": [
    {
      "plate_number": "ABC1234",
      "confidence": 0.87,
      "detected_at": "2025-01-07 10:30:00",
      "status": "registered"
    }
  ]
}
```

## Configuration

Edit `config.py` to customize:

```python
# Database
DATABASE_PATH = 'vehicle_registry.db'

# YOLO Model
YOLO_MODEL_PATH = None  # Or path to custom model

# Detection threshold
DETECTION_CONFIDENCE = 0.3

# OCR settings
OCR_LANGUAGES = ['en']
USE_GPU = False

# Server settings (in webapp.py)
PORT = 5000
HOST = '0.0.0.0'  # Access from network
DEBUG = True
```

## Access from Other Devices

By default, the server binds to `0.0.0.0`, making it accessible from other devices on your network.

1. Find your computer's IP address:
   ```bash
   # Windows
   ipconfig

   # Linux/Mac
   ifconfig
   ```

2. Access from other device:
   ```
   http://<your-ip-address>:5000
   ```

   Example: `http://192.168.1.100:5000`

## Project Structure (Web App)

```
Vehicle Registration System/
├── webapp.py                 # Flask application (main entry)
├── templates/
│   ├── base.html            # Base template
│   ├── dashboard.html       # Detection dashboard
│   ├── registry.html        # Vehicle registry
│   ├── detections.html      # Detection history
│   ├── 404.html            # Not found page
│   └── 500.html            # Error page
├── static/
│   ├── css/
│   │   └── style.css       # Styles
│   ├── js/
│   │   ├── main.js         # Common utilities
│   │   ├── dashboard.js    # Dashboard logic
│   │   ├── registry.js     # Registry logic
│   │   └── detections.js   # Detections logic
│   └── uploads/            # Uploaded images
├── services/                # Detection services
├── models/                  # Database models
└── config.py               # Configuration
```

## Desktop vs Web Application

| Feature | Desktop (app.py) | Web (webapp.py) |
|---------|-----------------|----------------|
| Interface | Tkinter GUI | Browser-based |
| Access | Local only | Network accessible |
| Multi-user | No | Yes |
| Mobile support | No | Yes |
| API | No | Yes |
| Real-time | Yes | Via refresh |

## Troubleshooting

### Port already in use
```bash
# Change port in webapp.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Can't access from network
- Check firewall settings
- Ensure host is '0.0.0.0'
- Verify network connectivity

### Uploads not working
- Check folder permissions
- Verify `static/uploads` directory exists
- Check file size (max 16MB)

### Slow performance
- First detection is slower (model loading)
- Enable GPU in config.py if available
- Reduce image size before upload

## Production Deployment

For production use, consider:

1. **Use production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 webapp:app
   ```

2. **Add authentication**:
   - Implement user login
   - Use Flask-Login

3. **Use proper database**:
   - PostgreSQL or MySQL
   - Instead of SQLite

4. **Enable HTTPS**:
   - Use SSL certificate
   - Nginx reverse proxy

5. **Set DEBUG=False**:
   - In production mode

## Testing the API

Using curl:

```bash
# Register a vehicle
curl -X POST http://localhost:5000/api/vehicles \
  -H "Content-Type: application/json" \
  -d '{
    "plate_number": "TEST123",
    "owner_name": "Test User",
    "vehicle_type": "Car"
  }'

# Get all vehicles
curl http://localhost:5000/api/vehicles

# Detect plate
curl -X POST http://localhost:5000/api/detect \
  -F "image=@/path/to/vehicle.jpg"
```

Using Python:

```python
import requests

# Register vehicle
response = requests.post('http://localhost:5000/api/vehicles', json={
    'plate_number': 'TEST123',
    'owner_name': 'Test User',
    'vehicle_type': 'Car'
})
print(response.json())

# Detect plate
with open('vehicle.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/detect',
                           files={'image': f})
print(response.json())
```

## Support

For issues or questions:
- Check the main README.md
- Review error messages in browser console
- Check Flask server logs in terminal

---

**Ready to go!** Just run `python webapp.py` and open http://localhost:5000 in your browser.
