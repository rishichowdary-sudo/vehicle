import cv2
import numpy as np
from ultralytics import YOLO
import os


class PlateDetector:
    """Detects license plates in images using YOLOv8."""

    def __init__(self, model_path=None, confidence_threshold=0.3):
        """Initialize plate detector.

        Args:
            model_path: Path to custom YOLO model (uses pretrained if None)
            confidence_threshold: Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold

        # Use pretrained YOLOv8 model or custom license plate model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Using YOLOv8n (nano) for faster inference
            # For license plate specific model, download from:
            # https://github.com/MuhammadMoinFaisal/YOLOv8-License-Plate-Detection
            self.model = YOLO('yolov8n.pt')

        print(f"PlateDetector initialized with confidence threshold: {confidence_threshold}")

    def detect_plates(self, image_path):
        """Detect license plates in an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of tuples: [(cropped_plate_image, bbox, confidence), ...]
        """
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return []

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return []

        # Run YOLO detection
        results = self.model(image, conf=self.confidence_threshold)

        detected_plates = []

        # Process each detection
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Crop the plate region
                plate_image = image[y1:y2, x1:x2]

                # Store detection
                detected_plates.append({
                    'image': plate_image,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        return detected_plates

    def detect_plates_from_array(self, image_array):
        """Detect license plates from numpy array (for UI integration).

        Args:
            image_array: numpy array of the image

        Returns:
            List of tuples: [(cropped_plate_image, bbox, confidence), ...]
        """
        if image_array is None or len(image_array) == 0:
            return []

        # Run YOLO detection
        results = self.model(image_array, conf=self.confidence_threshold)

        detected_plates = []

        # Process each detection
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Crop the plate region
                plate_image = image_array[y1:y2, x1:x2]

                # Store detection
                detected_plates.append({
                    'image': plate_image,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        return detected_plates

    def draw_detections(self, image_path, output_path=None):
        """Draw bounding boxes on detected plates and save/display.

        Args:
            image_path: Path to input image
            output_path: Path to save output image (shows if None)

        Returns:
            Annotated image array
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        detections = self.detect_plates(image_path)

        # Draw each detection
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add confidence label
            label = f'Plate: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save or show
        if output_path:
            cv2.imwrite(output_path, image)

        return image
