import cv2
import numpy as np
import easyocr
import re


class PlateOCR:
    """Reads text from license plate images using EasyOCR."""

    def __init__(self, languages=['en'], gpu=False):
        """Initialize OCR reader.

        Args:
            languages: List of languages to recognize (default: English)
            gpu: Use GPU acceleration if available
        """
        print("Initializing EasyOCR reader (this may take a moment on first run)...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR reader initialized successfully")

    def preprocess_plate(self, plate_image):
        """Preprocess plate image for better OCR accuracy.

        Args:
            plate_image: Cropped license plate image

        Returns:
            Preprocessed image
        """
        if plate_image is None or len(plate_image) == 0:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return thresh

    def read_plate(self, plate_image, preprocess=True):
        """Read text from a license plate image.

        Args:
            plate_image: Cropped license plate image
            preprocess: Apply preprocessing for better accuracy

        Returns:
            Dictionary with 'text', 'confidence', and 'raw_results'
        """
        if plate_image is None or len(plate_image) == 0:
            return {'text': '', 'confidence': 0.0, 'raw_results': []}

        # Preprocess if requested
        if preprocess:
            processed_image = self.preprocess_plate(plate_image)
        else:
            processed_image = plate_image

        # Run OCR
        results = self.reader.readtext(processed_image)

        if not results:
            # Try with original image if preprocessing didn't work
            if preprocess:
                results = self.reader.readtext(plate_image)

        if not results:
            return {'text': '', 'confidence': 0.0, 'raw_results': []}

        # Combine all detected text
        full_text = ' '.join([text for (bbox, text, conf) in results])

        # Calculate average confidence
        avg_confidence = np.mean([conf for (bbox, text, conf) in results]) if results else 0.0

        # Clean up the text
        cleaned_text = self.clean_plate_text(full_text)

        return {
            'text': cleaned_text,
            'confidence': float(avg_confidence),
            'raw_results': results
        }

    def clean_plate_text(self, text):
        """Clean and format license plate text.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned plate number
        """
        # Remove spaces and convert to uppercase
        text = text.upper().replace(' ', '')

        # Remove special characters (keep only alphanumeric)
        text = re.sub(r'[^A-Z0-9]', '', text)

        return text

    def read_multiple_plates(self, plate_images):
        """Read text from multiple plate images.

        Args:
            plate_images: List of cropped plate images

        Returns:
            List of dictionaries with OCR results
        """
        results = []

        for plate_image in plate_images:
            result = self.read_plate(plate_image)
            results.append(result)

        return results

    def read_with_fallback(self, plate_image):
        """Try multiple preprocessing methods for best accuracy.

        Args:
            plate_image: Cropped license plate image

        Returns:
            Best OCR result based on confidence
        """
        results = []

        # Try 1: With preprocessing
        result1 = self.read_plate(plate_image, preprocess=True)
        results.append(result1)

        # Try 2: Without preprocessing
        result2 = self.read_plate(plate_image, preprocess=False)
        results.append(result2)

        # Try 3: With different preprocessing - increase contrast
        enhanced = cv2.convertScaleAbs(plate_image, alpha=1.5, beta=10)
        result3 = self.read_plate(enhanced, preprocess=False)
        results.append(result3)

        # Return result with highest confidence
        best_result = max(results, key=lambda x: x['confidence'])
        return best_result
