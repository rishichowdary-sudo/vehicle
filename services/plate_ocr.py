import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging

# Suppress Paddle logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PlateOCR:
    """Reads text from license plate images using PaddleOCR."""

    def __init__(self, languages=['en'], gpu=False):
        """Initialize OCR reader.

        Args:
            languages: List of languages (kept for compatibility, Paddle uses 'en' by default)
            gpu: Use GPU acceleration if available
        """
        print("Initializing PaddleOCR reader...")
        # PaddleOCR handles model downloads automatically
        # Disabled angle classifier to prevent errors and improve speed
        self.reader = PaddleOCR(use_angle_cls=False, lang='en')
        print("PaddleOCR reader initialized successfully")

    def preprocess_plate(self, plate_image):
        """Preprocess plate image (Optional, Paddle is usually robust enough without)."""
        if plate_image is None or len(plate_image) == 0:
            return None
        
        # PaddleOCR handles preprocessing internally, but basic contrast enhancement can help
        # Keeping it simple as requested, but ensuring valid input
        return plate_image

    def read_plate(self, plate_image, preprocess=True):
        """Read text from a license plate image.

        Args:
            plate_image: Cropped license plate image
            preprocess: (Ignored for Paddle, kept for API compatibility)

        Returns:
            Dictionary with 'text', 'confidence', and 'raw_results'
        """
        if plate_image is None or len(plate_image) == 0:
            return {'text': '', 'confidence': 0.0, 'raw_results': []}

        try:
            # PaddleOCR expects RGB image (OpenCV gives BGR)
            if len(plate_image.shape) == 3:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)

            # PaddleOCR expects path or numpy array
            # simplified call due to argument errors in this environment
            results = self.reader.ocr(plate_image)
            
            print(f"DEBUG: OCR Raw Results: {results}")
            
            if not results or results[0] is None:
                return {'text': '', 'confidence': 0.0, 'raw_results': []}

            # Extract text and confidence
            detected_texts = []
            confidences = []

            # Robust parsing for various PaddleOCR result structures
            
            def extract_text_score(data):
                if isinstance(data, dict):
                    # Check for 'rec_texts' and 'rec_scores' keys (Newer PaddleX format?)
                    if 'rec_texts' in data and 'rec_scores' in data:
                        texts = data['rec_texts']
                        scores = data['rec_scores']
                        if isinstance(texts, list) and isinstance(scores, list):
                            for t, s in zip(texts, scores):
                                detected_texts.append(t)
                                confidences.append(s)
                    else:
                        # Recurse into values
                        for val in data.values():
                            extract_text_score(val)
                            
                elif isinstance(data, (list, tuple)):
                    # Check if it's a [text, score] pair
                    if len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], (float, int)):
                         detected_texts.append(data[0])
                         confidences.append(data[1])
                    else:
                        for item in data:
                            extract_text_score(item)
            
            extract_text_score(results)
            
            if not detected_texts:
                return {'text': '', 'confidence': 0.0, 'raw_results': results}

            # Combine text (plates might be multi-line or split)
            # Filter out low-confidence detections (noise removal)
            final_texts = []
            final_scores = []
            
            for t, s in zip(detected_texts, confidences):
                if s > 0.60:  # Minimum confidence threshold for character chunks
                    final_texts.append(t)
                    final_scores.append(s)
            
            if not final_texts:
                 # If everything was low confidence, fall back to the highest one (unsafe but better than empty)
                 if detected_texts:
                     max_idx = confidences.index(max(confidences))
                     final_texts.append(detected_texts[max_idx])
                     final_scores.append(confidences[max_idx])
            
            text = " ".join(final_texts)
            confidence = sum(final_scores) / len(final_scores) if final_scores else 0.0

            # Minimal cleaning: remove non-alphanumeric (but keep the raw read mostly)
            cleaned_text = self.clean_plate_text(text)

            return {
                'text': cleaned_text,
                'confidence': float(confidence),
                'raw_results': results
            }
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': '', 'confidence': 0.0, 'raw_results': []}

    def clean_plate_text(self, text):
        """Basic cleaning of plate text."""
        # Convert to uppercase
        text = text.upper()
        # Remove special characters but keep letters and numbers
        # This isn't a "safety net" logic (pattern matching), just standard string cleanup
        import re
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    def read_multiple_plates(self, plate_images):
        """Read text from multiple plate images."""
        results = []
        for plate_image in plate_images:
            result = self.read_plate(plate_image)
            results.append(result)
        return results

    def read_with_fallback(self, plate_image):
        """For Paddle, fallback is less necessary, but keeping method sig."""
        # Just call read_plate directly
        return self.read_plate(plate_image)

