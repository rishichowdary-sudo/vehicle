import cv2
import os
from services.plate_detector import PlateDetector
from services.plate_ocr import PlateOCR
from models.database import Database
from models.vehicle import Vehicle


class VehicleService:
    """Main service for vehicle plate detection and recognition."""

    def __init__(self, database=None, detector_model=None, use_gpu=False):
        """Initialize vehicle service.

        Args:
            database: Database instance (creates new if None)
            detector_model: Path to custom YOLO model
            use_gpu: Use GPU for OCR if available
        """
        print("Initializing Vehicle Service...")

        # Initialize database
        self.db = database if database else Database()

        # Initialize plate detector
        print("Loading plate detector...")
        self.detector = PlateDetector(model_path=detector_model)

        # Initialize OCR
        print("Loading OCR engine...")
        self.ocr = PlateOCR(gpu=use_gpu)

        print("Vehicle Service initialized successfully!")

    def process_image(self, image_path, log_detection=True):
        """Process an image to detect and read license plates.

        Args:
            image_path: Path to the image file
            log_detection: Whether to log detection to database

        Returns:
            Dictionary with detection results
        """
        results = {
            'success': False,
            'detections': [],
            'error': None
        }

        try:
            # Step 1: Detect plates
            detected_plates = self.detector.detect_plates(image_path)

            if not detected_plates:
                results['error'] = "No license plates detected in image"
                return results

            # Step 2: Read text from each detected plate
            for detection in detected_plates:
                plate_image = detection['image']
                bbox = detection['bbox']
                det_confidence = detection['confidence']

                # Run OCR
                ocr_result = self.ocr.read_with_fallback(plate_image)

                plate_number = ocr_result['text']
                ocr_confidence = ocr_result['confidence']

                # Check if vehicle is registered
                vehicle = self.db.get_vehicle_by_plate(plate_number) if plate_number else None

                detection_info = {
                    'plate_number': plate_number,
                    'detection_confidence': det_confidence,
                    'ocr_confidence': ocr_confidence,
                    'bbox': bbox,
                    'registered': vehicle is not None,
                    'vehicle_info': vehicle.to_dict() if vehicle else None
                }

                results['detections'].append(detection_info)

                # Log to database
                if log_detection and plate_number:
                    status = 'registered' if vehicle else 'unregistered'
                    self.db.log_detection(
                        plate_number=plate_number,
                        confidence=ocr_confidence,
                        image_path=image_path,
                        status=status
                    )

            results['success'] = True
            return results

        except Exception as e:
            results['error'] = str(e)
            return results

    def process_image_array(self, image_array, log_detection=False):
        """Process an image array (from UI) to detect and read plates.

        Args:
            image_array: numpy array of the image
            log_detection: Whether to log detection

        Returns:
            Dictionary with detection results
        """
        results = {
            'success': False,
            'detections': [],
            'error': None
        }

        try:
            # Step 1: Detect plates
            detected_plates = self.detector.detect_plates_from_array(image_array)

            if not detected_plates:
                results['error'] = "No license plates detected"
                return results

            # Step 2: Read text from each detected plate
            for detection in detected_plates:
                plate_image = detection['image']
                bbox = detection['bbox']
                det_confidence = detection['confidence']

                # Run OCR
                ocr_result = self.ocr.read_with_fallback(plate_image)

                plate_number = ocr_result['text']
                ocr_confidence = ocr_result['confidence']

                # Check if vehicle is registered
                vehicle = self.db.get_vehicle_by_plate(plate_number) if plate_number else None

                detection_info = {
                    'plate_number': plate_number,
                    'detection_confidence': det_confidence,
                    'ocr_confidence': ocr_confidence,
                    'bbox': bbox,
                    'registered': vehicle is not None,
                    'vehicle_info': vehicle.to_dict() if vehicle else None
                }

                results['detections'].append(detection_info)

                # Log to database
                if log_detection and plate_number:
                    status = 'registered' if vehicle else 'unregistered'
                    self.db.log_detection(
                        plate_number=plate_number,
                        confidence=ocr_confidence,
                        status=status
                    )

            results['success'] = True
            return results

        except Exception as e:
            results['error'] = str(e)
            return results

    def register_vehicle(self, plate_number, owner_name, vehicle_type=None,
                        color=None, model=None):
        """Register a new vehicle in the system.

        Args:
            plate_number: License plate number
            owner_name: Name of the owner
            vehicle_type: Type of vehicle
            color: Vehicle color
            model: Vehicle make/model

        Returns:
            True if successful, False otherwise
        """
        return self.db.add_vehicle(
            plate_number=plate_number,
            owner_name=owner_name,
            vehicle_type=vehicle_type,
            color=color,
            model=model
        )

    def get_vehicle_info(self, plate_number):
        """Get information about a registered vehicle.

        Args:
            plate_number: License plate number

        Returns:
            Vehicle object if found, None otherwise
        """
        return self.db.get_vehicle_by_plate(plate_number)

    def get_all_vehicles(self):
        """Get all registered vehicles.

        Returns:
            List of Vehicle objects
        """
        return self.db.get_all_vehicles()

    def update_vehicle_info(self, plate_number, **kwargs):
        """Update vehicle information.

        Args:
            plate_number: Plate number to update
            **kwargs: Fields to update

        Returns:
            True if successful, False otherwise
        """
        return self.db.update_vehicle(plate_number, **kwargs)

    def delete_vehicle(self, plate_number):
        """Delete a vehicle from the system.

        Args:
            plate_number: Plate number to delete

        Returns:
            True if successful, False otherwise
        """
        return self.db.delete_vehicle(plate_number)

    def get_recent_detections(self, limit=50):
        """Get recent detection logs.

        Args:
            limit: Maximum number of records

        Returns:
            List of detection records
        """
        return self.db.get_recent_detections(limit)

    def get_stats(self):
        """Get system statistics.

        Returns:
            Dictionary with system stats
        """
        return {
            'total_vehicles': self.db.get_vehicle_count(),
            'recent_detections': len(self.db.get_recent_detections(10))
        }
