# Vehicle Plate Detection & Registration System

A comprehensive license plate detection and recognition system using **YOLOv8** for plate detection and **EasyOCR** for text recognition. Built with Python and Tkinter.

## Features

- **Automatic License Plate Detection**: Uses YOLOv8 deep learning model to detect license plates in images
- **OCR Text Recognition**: Reads plate numbers using EasyOCR with preprocessing
- **Vehicle Registration Database**: SQLite database to store and manage registered vehicles
- **Detection Logging**: Tracks all plate detections with timestamps
- **User-Friendly GUI**: Tkinter-based interface with two main tabs:
  - Detection Dashboard: Upload images and detect plates
  - Vehicle Registry: Manage registered vehicles
- **Visual Feedback**: Annotated images with bounding boxes (green for registered, red for unregistered)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or navigate to the project directory**:
   ```bash
   cd "C:\Users\Rishichowdary-3925\Downloads\Vehicle Registration System"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - OpenCV for image processing
   - YOLOv8 (ultralytics) for plate detection
   - EasyOCR for text recognition
   - PyTorch for deep learning
   - Other required packages

   **Note**: First installation may take several minutes as it downloads pre-trained models.

3. **Run the application**:
   ```bash
   python app.py
   ```

## Usage

### Detection Dashboard

1. Click **"Load Image"** to select a vehicle image
2. Click **"Detect Plate"** to process the image
3. Results will show:
   - Detected plate number
   - Detection confidence
   - OCR confidence
   - Registration status
4. If unregistered, the plate number auto-fills in the registration form
5. Use **"Quick Registration"** section to register new vehicles

### Vehicle Registry

1. View all registered vehicles in a table
2. Search vehicles by plate number or owner name
3. Add new vehicles manually
4. Delete vehicles
5. Double-click to view detailed information

## Project Structure

```
Vehicle Registration System/
├── services/
│   ├── plate_detector.py    # YOLOv8 plate detection
│   ├── plate_ocr.py          # EasyOCR text recognition
│   └── vehicle_service.py    # Main service combining detection + OCR
├── models/
│   ├── vehicle.py            # Vehicle data model
│   └── database.py           # SQLite database operations
├── ui/
│   ├── dashboard.py          # Detection dashboard UI
│   ├── registry.py           # Vehicle registry UI
│   └── main_window.py        # Main application window
├── config.py                 # Configuration settings
├── app.py                    # Main entry point
└── requirements.txt          # Dependencies
```

## Configuration

Edit `config.py` to customize:

- **YOLO_MODEL_PATH**: Path to custom plate detection model
- **DETECTION_CONFIDENCE**: Confidence threshold (0.0-1.0)
- **USE_GPU**: Enable GPU acceleration if available
- **OCR_LANGUAGES**: Languages for OCR recognition

## Using a Custom License Plate Model

For better accuracy with specific license plate formats:

1. Download a license plate specific YOLOv8 model:
   - [YOLOv8 License Plate Detection](https://github.com/MuhammadMoinFaisal/YOLOv8-License-Plate-Detection)
   - Or train your own on Roboflow

2. Update `config.py`:
   ```python
   YOLO_MODEL_PATH = 'path/to/your/model.pt'
   ```

## How It Works

### Detection Pipeline

1. **Image Input**: User uploads a vehicle image
2. **Plate Detection**: YOLOv8 identifies plate regions in the image
3. **Preprocessing**: Applies grayscale, filtering, and thresholding
4. **OCR**: EasyOCR reads text from detected plate regions
5. **Database Lookup**: Checks if plate is registered
6. **Display Results**: Shows annotated image and vehicle information

### Similar to Facial Recognition

Just like your facial recognition system uses:
- **InsightFace** for face detection and embeddings
- **Cosine similarity** for matching faces
- **Database** for storing employee data

This system uses:
- **YOLOv8** for plate detection
- **EasyOCR** for reading text
- **Database** for storing vehicle data

## Testing

Create a simple test script:

```python
from services.vehicle_service import VehicleService

# Initialize service
service = VehicleService()

# Register a test vehicle
service.register_vehicle(
    plate_number="ABC1234",
    owner_name="John Doe",
    vehicle_type="Car"
)

# Process an image
results = service.process_image("path/to/vehicle.jpg")
print(results)
```

## Troubleshooting

### Issue: "No module named 'tkinter'"
- **Windows**: Reinstall Python with "tcl/tk and IDLE" option checked
- **Linux**: `sudo apt-get install python3-tk`
- **Mac**: Tkinter should be included with Python

### Issue: Models downloading slowly
- First run downloads YOLOv8 and EasyOCR models (100-500MB)
- Subsequent runs will use cached models

### Issue: Low accuracy
- Try adjusting `DETECTION_CONFIDENCE` in config.py
- Use a custom license plate specific YOLO model
- Ensure input images are clear and well-lit

## GPU Acceleration

For faster processing with NVIDIA GPU:

1. Install CUDA Toolkit
2. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Set in `config.py`:
   ```python
   USE_GPU = True
   ```

## Database

- **Location**: `vehicle_registry.db` (SQLite)
- **Tables**:
  - `vehicles`: Registered vehicles
  - `detections`: Detection logs

## License

This is a demonstration project for educational purposes.

## Credits

- **YOLOv8**: Ultralytics
- **EasyOCR**: JaidedAI
- **Similar Architecture**: Based on the facial recognition prototype
