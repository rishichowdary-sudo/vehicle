"""
License Plate Detection Alternatives Comparison
Compare actual plate detection methods (not general object detectors)
"""

import os
import sys
import cv2
import time
from pathlib import Path
import json
from datetime import datetime
import numpy as np

sys.path.append(os.path.dirname(__file__))
from services.plate_ocr import PlateOCR


class PlateDetectionComparison:
    """Compare different plate detection approaches."""

    def __init__(self, test_data_path):
        self.test_data_path = Path(test_data_path)
        self.ocr = PlateOCR()
        self.test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.test_images.extend(list(self.test_data_path.glob(ext)))
        print(f"Found {len(self.test_images)} test images")
        self.results = {}

    def test_custom_yolo(self):
        """Test custom YOLO-based plate detector (baseline)."""
        print(f"\n{'='*60}")
        print(f"Testing: Custom YOLO Plate Detector (Current)")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLO
            detector = YOLO('license_plate_detector.pt')

            metrics = self._run_detection_pipeline(
                detector,
                'custom_yolo',
                lambda img: self._yolo_detect(detector, img)
            )

            self.results['Custom YOLO'] = metrics
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def test_haar_cascade(self):
        """Test Haar Cascade plate detection (Traditional CV)."""
        print(f"\n{'='*60}")
        print(f"Testing: Haar Cascade (Traditional CV)")
        print(f"{'='*60}")

        try:
            # Try to load plate cascade
            cascade_paths = [
                'haarcascade_russian_plate_number.xml',
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            ]

            cascade = None
            for path in cascade_paths:
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    break

            if cascade is None or cascade.empty():
                print("[INFO] Downloading Haar Cascade for plates...")
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
                cascade_path = "haarcascade_plate.xml"
                urllib.request.urlretrieve(url, cascade_path)
                cascade = cv2.CascadeClassifier(cascade_path)

            if cascade.empty():
                print("[SKIP] Could not load Haar Cascade")
                return None

            metrics = self._run_detection_pipeline(
                cascade,
                'haar',
                lambda img: self._haar_detect(cascade, img)
            )

            self.results['Haar Cascade'] = metrics
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_contour_based(self):
        """Test contour-based plate detection (Classical CV)."""
        print(f"\n{'='*60}")
        print(f"Testing: Contour-Based Detection (Classical CV)")
        print(f"{'='*60}")

        try:
            metrics = self._run_detection_pipeline(
                None,
                'contour',
                lambda img: self._contour_detect(img)
            )

            self.results['Contour-Based'] = metrics
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_wpod_net(self):
        """Test WPOD-NET (Warped Planar Object Detection Network)."""
        print(f"\n{'='*60}")
        print(f"Testing: WPOD-NET (Deep Learning)")
        print(f"{'='*60}")

        try:
            # Check if keras/tensorflow available
            import tensorflow as tf
            from tensorflow import keras

            # Try to load or download WPOD-NET model
            model_path = "wpod-net.h5"

            if not os.path.exists(model_path):
                print("[INFO] WPOD-NET model not found")
                print("[SKIP] Download from: https://github.com/sergiomsilva/alpr-unconstrained")
                return None

            model = keras.models.load_model(model_path, compile=False)

            metrics = self._run_detection_pipeline(
                model,
                'wpod',
                lambda img: self._wpod_detect(model, img)
            )

            self.results['WPOD-NET'] = metrics
            return metrics

        except ImportError:
            print("[SKIP] TensorFlow/Keras not installed")
            return None
        except Exception as e:
            print(f"[SKIP] {e}")
            return None

    def _yolo_detect(self, detector, img):
        """YOLO detection method."""
        results_list = detector(img, conf=0.3, verbose=False)
        detections = []

        for result in results_list:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Add padding
                h, w = img.shape[:2]
                pad_x = int((x2-x1) * 0.2)
                pad_y = int((y2-y1) * 0.05)
                x1 = max(0, x1-pad_x)
                y1 = max(0, y1-pad_y)
                x2 = min(w, x2+pad_x)
                y2 = min(h, y2+pad_y)

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf
                })

        return detections

    def _haar_detect(self, cascade, img):
        """Haar Cascade detection method."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 10)
        )

        detections = []
        for (x, y, w, h) in plates:
            # Add padding
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)

            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': 0.9  # Haar doesn't provide confidence
            })

        return detections

    def _contour_detect(self, img):
        """Contour-based detection method."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small contours
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0

            # Plate aspect ratio typically 2:1 to 5:1
            if 2.0 <= aspect_ratio <= 6.0:
                # Add padding
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': min(aspect_ratio / 5.0, 1.0)
                })

        return detections

    def _wpod_detect(self, model, img):
        """WPOD-NET detection method."""
        # Resize image
        target_size = (288, 288)
        resized = cv2.resize(img, target_size)
        resized = resized / 255.0
        resized = np.expand_dims(resized, axis=0)

        # Predict
        predictions = model.predict(resized, verbose=0)

        # Process predictions (simplified)
        detections = []
        # Note: Full WPOD-NET post-processing is complex
        # This is a placeholder
        return detections

    def _run_detection_pipeline(self, detector, method_name, detect_func):
        """Run detection + OCR pipeline."""
        total_time_det = 0
        total_time_ocr = 0
        total_detections = 0
        successful_reads = 0
        det_confs = []
        ocr_confs = []
        results = []
        images_with_detection = 0

        for img_path in self.test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detection
            start = time.time()
            detections = detect_func(img)
            det_time = time.time() - start
            total_time_det += det_time

            if detections:
                images_with_detection += 1

            # Process each detection
            for det in detections:
                total_detections += 1
                x1, y1, x2, y2 = det['box']
                conf = det['confidence']
                det_confs.append(conf)

                plate = img[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                # OCR
                start = time.time()
                ocr_result = self.ocr.read_plate(plate)
                ocr_time = time.time() - start
                total_time_ocr += ocr_time

                text = ocr_result.get('text', '')
                ocr_conf = ocr_result.get('confidence', 0)

                if text and len(text) > 3:
                    successful_reads += 1
                    ocr_confs.append(ocr_conf)
                    results.append({
                        'image': img_path.name,
                        'text': text,
                        'det_conf': conf,
                        'ocr_conf': ocr_conf
                    })
                    print(f"  [OK] {img_path.name}: {text} (Det:{conf:.2f}, OCR:{ocr_conf:.2f})")

        # Calculate metrics
        num_images = len(self.test_images)
        metrics = {
            'total_detections': total_detections,
            'successful_reads': successful_reads,
            'images_with_detection': images_with_detection,
            'detection_rate': (images_with_detection / num_images) * 100,
            'avg_det_time_ms': (total_time_det / num_images) * 1000,
            'avg_ocr_time_ms': (total_time_ocr / total_detections) * 1000 if total_detections > 0 else 0,
            'avg_total_time_ms': ((total_time_det / num_images) + (total_time_ocr / total_detections if total_detections > 0 else 0)) * 1000,
            'avg_det_conf': np.mean(det_confs) if det_confs else 0,
            'avg_ocr_conf': np.mean(ocr_confs) if ocr_confs else 0,
            'ocr_success_rate': (successful_reads / total_detections) * 100 if total_detections > 0 else 0,
            'end_to_end_success': (successful_reads / num_images) * 100,
            'sample_results': results[:10]
        }

        print(f"\nSummary:")
        print(f"  Detection Rate: {metrics['detection_rate']:.1f}% ({images_with_detection}/{num_images})")
        print(f"  Plates Detected: {total_detections}")
        print(f"  OCR Success: {metrics['ocr_success_rate']:.1f}% ({successful_reads}/{total_detections})")
        print(f"  End-to-End: {metrics['end_to_end_success']:.1f}% ({successful_reads}/{num_images})")

        return metrics

    def generate_report(self):
        """Generate comparison report."""
        if not self.results:
            print("No results available")
            return

        print("\n" + "="*60)
        print("PLATE DETECTION ALTERNATIVES COMPARISON")
        print("="*60)

        # Sort by end-to-end success
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('end_to_end_success', 0),
            reverse=True
        )

        print("\n1. END-TO-END SUCCESS (Detection + OCR):")
        print("-"*60)
        for i, (name, m) in enumerate(sorted_results, 1):
            success = m.get('end_to_end_success', 0)
            reads = m.get('successful_reads', 0)
            total = len(self.test_images)
            time_ms = m.get('avg_total_time_ms', 0)
            print(f"{i}. {name:25s} {success:5.1f}% ({reads}/{total}) | {time_ms:6.1f}ms")

        print("\n2. DETECTION RATE:")
        print("-"*60)
        sorted_det = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('detection_rate', 0),
            reverse=True
        )
        for i, (name, m) in enumerate(sorted_det, 1):
            rate = m.get('detection_rate', 0)
            imgs = m.get('images_with_detection', 0)
            print(f"{i}. {name:25s} {rate:5.1f}% ({imgs}/{len(self.test_images)} images)")

        print("\n3. OCR SUCCESS RATE (on detected plates):")
        print("-"*60)
        sorted_ocr = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('ocr_success_rate', 0),
            reverse=True
        )
        for i, (name, m) in enumerate(sorted_ocr, 1):
            rate = m.get('ocr_success_rate', 0)
            reads = m.get('successful_reads', 0)
            total = m.get('total_detections', 0)
            print(f"{i}. {name:25s} {rate:5.1f}% ({reads}/{total} plates)")

        print("\n4. DETECTION SPEED:")
        print("-"*60)
        sorted_speed = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('avg_det_time_ms', float('inf'))
        )
        for i, (name, m) in enumerate(sorted_speed, 1):
            time_ms = m.get('avg_det_time_ms', 0)
            print(f"{i}. {name:25s} {time_ms:6.1f}ms")

        print("\n5. DETECTION CONFIDENCE:")
        print("-"*60)
        sorted_conf = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('avg_det_conf', 0),
            reverse=True
        )
        for i, (name, m) in enumerate(sorted_conf, 1):
            conf = m.get('avg_det_conf', 0)
            print(f"{i}. {name:25s} {conf:.3f}")

        # Recommendation
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)

        if sorted_results:
            winner = sorted_results[0]
            print(f"\nBest Overall: {winner[0]}")
            print(f"  - End-to-end success: {winner[1]['end_to_end_success']:.1f}%")
            print(f"  - OCR success rate: {winner[1]['ocr_success_rate']:.1f}%")
            print(f"  - Total time: {winner[1]['avg_total_time_ms']:.1f}ms")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"plate_detection_comparison_{timestamp}.json"

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_images': len(self.test_images),
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved: {report_file}")
        print("="*60)


def main():
    """Main execution."""
    print("="*60)
    print("LICENSE PLATE DETECTION ALTERNATIVES")
    print("="*60)

    test_path = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_path):
        print(f"[ERROR] Path not found: {test_path}")
        return

    comparison = PlateDetectionComparison(test_path)

    # Test all methods
    comparison.test_custom_yolo()
    comparison.test_haar_cascade()
    comparison.test_contour_based()
    comparison.test_wpod_net()

    # Generate report
    comparison.generate_report()


if __name__ == "__main__":
    main()
