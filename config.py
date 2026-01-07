# Configuration for Vehicle Plate Detection System

# Database settings
DATABASE_PATH = 'vehicle_registry.db'

# YOLO Model settings
# YOLO Model settings
YOLO_MODEL_PATH = 'license_plate_detector.pt'  # Custom license plate detector

# Detection confidence threshold (0.0 - 1.0)
DETECTION_CONFIDENCE = 0.3

# OCR settings
OCR_LANGUAGES = ['en']
USE_GPU = False  # Set to True if you have CUDA-capable GPU

# UI settings
WINDOW_TITLE = "Vehicle Plate Detection & Registration System"
WINDOW_SIZE = "1200x700"
