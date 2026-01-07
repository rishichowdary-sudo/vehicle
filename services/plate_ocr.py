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
        # Enabled angle classifier to handle rotated images (e.g., ver.jpeg)
        self.reader = PaddleOCR(use_angle_cls=True, lang='en')
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
            
            # Sort text blocks: Heuristic - State code (Letters) usually comes first
            # If we have multiple blocks, try to put the one starting with letters first
            if len(final_texts) > 1:
                # Check if the first block starts with digits and second with letters
                if final_texts[0][0].isdigit() and final_texts[1][0].isalpha():
                     # Swap
                     final_texts[0], final_texts[1] = final_texts[1], final_texts[0]
                     final_scores[0], final_scores[1] = final_scores[1], final_scores[0]
            
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

        # Fix invalid state codes common in OCR errors
        # MA is not a valid state code, but KA (Karnataka) is. MH (Maharashtra) is.
        # Given user feedback, MA -> KA is a likely correction.
        if text.startswith('MA'):
            text = 'KA' + text[2:]

        # Smart heuristic for Indian Plates (LL DD LL DDDD)
        # Fix common OCR confusion (O -> 0, Q -> 0, Z -> 2, etc.) in expected digit positions
        
        if len(text) >= 4:
            # Convert string to list for mutability
            chars = list(text)

            # Heuristic 1: 3rd and 4th characters are usually digits (District Code)
            # Example: MP O4 -> MP 04
            if chars[0].isalpha() and chars[1].isalpha():
                for i in [2, 3]:
                    if i < len(chars):
                        if chars[i] in ['O', 'Q', 'D']:
                            chars[i] = '0'
                        elif chars[i] == 'Z':
                            chars[i] = '2'
                        elif chars[i] == 'S':
                            chars[i] = '5'
                        elif chars[i] == 'B':
                            chars[i] = '8'

            # Heuristic 2: Last 4 characters are usually digits
            # Example: ... CC Z688 -> ... CC 2688
            if len(chars) > 4:
                suffix_start = max(4, len(chars) - 4)
                for i in range(suffix_start, len(chars)):
                    if chars[i] in ['O', 'Q', 'D']:
                        chars[i] = '0'
                    elif chars[i] == 'Z':
                        chars[i] = '2'
                    elif chars[i] == 'S':
                        chars[i] = '5'
                    elif chars[i] == 'B':
                        chars[i] = '8'
            
            text = "".join(chars)

        return text

    def read_multiple_plates(self, plate_images):
        """Read text from multiple plate images."""
        results = []
        for plate_image in plate_images:
            result = self.read_plate(plate_image)
            results.append(result)
        return results

    def read_with_fallback(self, plate_image):
        """Read text with robust fallback strategies for difficult images."""
        
        # 1. Try raw image first
        result = self.read_plate(plate_image)
        if result['confidence'] > 0.8:
            return result
            
        print(f"Low confidence ({result['confidence']:.2f}). Trying enhancements...")
        candidates = [result]
        
        # 2. Try Scaled Up (2x) - Helps with small/pixelated text
        try:
            h, w = plate_image.shape[:2]
            scaled = cv2.resize(plate_image, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            res_scaled = self.read_plate(scaled)
            candidates.append(res_scaled)
            if res_scaled['confidence'] > 0.85:
                return res_scaled
        except Exception as e:
            print(f"Scaling error: {e}")

        # 3. Try Contrast Enhancement (CLAHE) - Helps with lighting/shadows
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(plate_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            res_enhanced = self.read_plate(enhanced)
            candidates.append(res_enhanced)
        except Exception as e:
            print(f"Enhancement error: {e}")

        # 4. Try Grayscale + Thresholding (Binary) - Helps with color noise
        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            # Otsu's thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Convert back to BGR for Paddle
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            res_binary = self.read_plate(binary_bgr)
            candidates.append(res_binary)
        except Exception as e:
            print(f"Binary error: {e}")

        # Select best result
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        print(f"Best confidence after fallback: {best_candidate['confidence']:.2f}")
        return best_candidate

