# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First installation will take 5-10 minutes as it downloads:
- YOLOv8 model (~50MB)
- EasyOCR models (~100-200MB)
- PyTorch and dependencies

## 2. Run the Application

```bash
python app.py
```

## 3. Basic Workflow

### Register a Vehicle

**Option A: From Detection Dashboard**
1. Click "Load Image" and select a vehicle image
2. Click "Detect Plate"
3. If plate is detected and unregistered, it auto-fills the form
4. Enter owner name and click "Register Vehicle"

**Option B: From Vehicle Registry Tab**
1. Switch to "Vehicle Registry" tab
2. Click "Add Vehicle"
3. Enter plate number, owner name, and other details
4. Click "Save"

### Detect Plates in Images

1. Go to "Detection Dashboard" tab
2. Click "Load Image" and select a vehicle image
3. Click "Detect Plate"
4. Results show:
   - Plate number (if readable)
   - Detection confidence
   - OCR confidence
   - Registration status (GREEN = registered, RED = unregistered)

## 4. Test Without GUI

Run the test script:

```bash
python test_detection.py
```

This will:
- Register sample vehicles
- List all vehicles
- Test plate detection (if you provide an image)

## 5. Tips for Best Results

### Image Quality
- Use clear, well-lit images
- Plate should be visible and not too small
- Avoid heavy shadows or reflections

### Supported Formats
- JPG, JPEG, PNG, BMP
- Recommended resolution: 640x480 or higher

### For Better Accuracy
1. **Use a custom license plate model**:
   - Download from: https://github.com/MuhammadMoinFaisal/YOLOv8-License-Plate-Detection
   - Update `YOLO_MODEL_PATH` in `config.py`

2. **Adjust confidence threshold**:
   - Lower `DETECTION_CONFIDENCE` in `config.py` if plates are missed
   - Raise it if too many false positives

## 6. Common Issues

### "No plates detected"
- Image quality may be poor
- Plate too small in image
- Try lowering `DETECTION_CONFIDENCE` in config.py

### OCR reading wrong text
- Plate image is blurry or skewed
- Unusual font or format
- Try using a license plate specific YOLO model

### Slow performance
- First detection is always slower (model loading)
- Enable GPU acceleration in config.py if you have NVIDIA GPU
- Use smaller images (resize to 1920x1080 or lower)

## 7. Sample Code

### Detect plate programmatically:

```python
from services.vehicle_service import VehicleService

# Initialize
service = VehicleService()

# Register a vehicle
service.register_vehicle("ABC1234", "John Doe", "Car")

# Detect from image
results = service.process_image("path/to/image.jpg")

if results['success']:
    for det in results['detections']:
        print(f"Plate: {det['plate_number']}")
        print(f"Registered: {det['registered']}")
```

## 8. Where to Get Test Images

### Free Sources:
1. Google Images: Search "license plate" or "car with plate"
2. Unsplash: https://unsplash.com/s/photos/license-plate
3. Pexels: https://www.pexels.com/search/car/

### Create Your Own:
- Take photos of vehicles in parking lots
- Use your own vehicle (ensure plate is visible)

## Need Help?

Check `README.md` for detailed documentation.
