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

            # Extract text with bounding box info for smart filtering
            text_blocks = []  # List of (text, score, area, y_pos, x_pos)

            def extract_with_boxes(data):
                """Extract text, scores, bounding box areas and positions from OCR results."""
                if isinstance(data, dict):
                    if 'rec_texts' in data and 'rec_scores' in data:
                        texts = data.get('rec_texts', [])
                        scores = data.get('rec_scores', [])
                        polys = data.get('rec_polys', [])

                        for i, (t, s) in enumerate(zip(texts, scores)):
                            height = 0
                            if i < len(polys) and len(polys[i]) > 0:
                                try:
                                    poly = polys[i]
                                    if hasattr(poly, 'shape'):
                                        xs = poly[:, 0]
                                        ys = poly[:, 1]
                                        width = max(xs) - min(xs)
                                        height = max(ys) - min(ys)
                                        area = width * height
                                        y_pos = min(ys)  # Top of bounding box
                                        x_pos = min(xs)  # Left of bounding box
                                except:
                                    area = len(t) * 100
                                    height = 30 # Default fallback
                            else:
                                area = len(t) * 100
                                height = 30
                            text_blocks.append((t, s, area, y_pos, x_pos, height))
                    else:
                        for val in data.values():
                            extract_with_boxes(val)
                elif isinstance(data, (list, tuple)):
                    if len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], (float, int)):
                        text_blocks.append((data[0], data[1], len(data[0]) * 100, 0, 0, 30))
                    else:
                        for item in data:
                            extract_with_boxes(item)

            extract_with_boxes(results)

            if not text_blocks:
                return {'text': '', 'confidence': 0.0, 'raw_results': results}

            # Filter and find the best plate number
            import re

            # Indian state codes
            states = {'AN','AP','AR','AS','BR','CG','CH','DD','DL','GA','GJ','HP','HR',
                     'JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD',
                     'PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB'}

            def is_valid_plate(text):
                """Check if text matches Indian plate pattern."""
                clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(clean) < 8 or len(clean) > 12:
                    return False
                # Must start with valid state code
                if clean[:2] not in states:
                    return False
                # Must have digits after state code
                if not clean[2:4].replace('O','0').replace('D','0').isdigit():
                    return False
                return True

            # Debug: show positions
            print(f"DEBUG: Raw blocks with positions: {[(t, y, x) for t, s, a, y, x, h in text_blocks]}")

            # Sort by Y position (top to bottom), then X (left to right)
            # Group by similar Y values (same line) first
            if text_blocks:
                # Calculate dynamic threshold based on average text height
                heights = [b[5] for b in text_blocks if b[5] > 0]
                avg_height = sum(heights) / len(heights) if heights else 30
                line_threshold = avg_height * 0.7  # 70% of character height logic
                print(f"DEBUG: Line threshold: {line_threshold:.2f} (Avg Height: {avg_height:.2f})")

                # Sort by Y first
                text_blocks.sort(key=lambda x: x[3])

                # Group blocks on same line
                lines = []
                current_line = [text_blocks[0]]
                for block in text_blocks[1:]:
                    if abs(block[3] - current_line[0][3]) < line_threshold:
                        current_line.append(block)
                    else:
                        lines.append(current_line)
                        current_line = [block]
                lines.append(current_line)

                # Sort each line by X (left to right)
                sorted_blocks = []
                for line in lines:
                    line.sort(key=lambda x: x[4])
                    sorted_blocks.extend(line)
                text_blocks = sorted_blocks

            # Combine all blocks in reading order
            all_texts = [re.sub(r'[^A-Z0-9]', '', t.upper()) for t, s, a, y, x, h in text_blocks]
            all_scores = [s for t, s, a, y, x, h in text_blocks]

            combined = ''.join(all_texts)
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

            print(f"DEBUG: Sorted blocks: {all_texts} -> {combined}")

            best_plate = None
            best_score = 0

            # Check if combined text is valid plate
            if is_valid_plate(combined):
                best_plate = combined
                best_score = avg_score
            else:
                # Try each single block
                for text, score in zip(all_texts, all_scores):
                    if is_valid_plate(text):
                        best_plate = text
                        best_score = score
                        break

            # Fallback: use largest text block
            if not best_plate:
                for block in text_blocks:
                    # Robust unpacking by index
                    text = block[0]
                    score = block[1]
                    # area = block[2]
                    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean) >= 6:
                        best_plate = clean
                        best_score = score
                        break

            text = best_plate or ''
            confidence = best_score

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
        import re
        text = text.upper()
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

