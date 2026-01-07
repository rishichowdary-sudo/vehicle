import os
import sys
import logging

# Setup basic logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vehicle_service import VehicleService
from models.database import Database
import config

def verify_image():
    image_path = r"C:\Users\Rishichowdary-3925\Downloads\bg.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Processing image: {image_path}")
    
    # Initialize service
    print("Initializing services...")
    try:
        # Use config settings
        db = Database(config.DATABASE_PATH)
        service = VehicleService(database=db, detector_model=config.YOLO_MODEL_PATH, use_gpu=config.USE_GPU)
        
        # Process
        print("Running detection...")
        results = service.process_image(image_path, log_detection=False)
        
        if results['success']:
            print("\n" + "="*50)
            print("DETECTION RESULTS")
            print("="*50)
            for i, det in enumerate(results['detections']):
                print(f"Detection {i+1}:")
                print(f"  Plate Number: {det['plate_number']}")
                print(f"  OCR Confidence: {det['ocr_confidence']:.2f}")
                print(f"  Detection Confidence: {det['detection_confidence']:.2f}")
                print(f"  Registered: {det['registered']}")
            print("="*50 + "\n")
        else:
            print(f"Detection failed: {results.get('error')}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Redirect stdout to a file for reliable capture by agent
    import sys
    original_stdout = sys.stdout
    with open('verification_output.txt', 'w') as f:
        sys.stdout = f
        verify_image()
    sys.stdout = original_stdout
    # Also print to console
    with open('verification_output.txt', 'r') as f:
        print(f.read())
