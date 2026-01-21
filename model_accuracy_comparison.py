"""
Model Accuracy Comparison Script
Compares different detection and OCR models for license plate recognition.
"""

import os
import cv2
import time
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
import logging

# Suppress unnecessary logs
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


class ModelComparison:
    """Compare different models for plate detection and OCR."""

    def __init__(self, test_data_path):
        """Initialize comparison with test data path."""
        self.test_data_path = Path(test_data_path)
        self.results = {
            'detection_models': {},
            'ocr_models': {},
            'combined_results': {}
        }

        # Get all test images
        self.test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.test_images.extend(list(self.test_data_path.glob(ext)))

        print(f"Found {len(self.test_images)} test images")

    def test_detection_models(self):
        """Test different YOLO models for plate detection."""
        print("\n" + "="*60)
        print("TESTING DETECTION MODELS")
        print("="*60)

        # Models to test (ordered from fastest to most accurate)
        detection_models = {
            'YOLOv8n': 'yolov8n.pt',
            'YOLOv8s': 'yolov8s.pt',
            'YOLOv8m': 'yolov8m.pt',
            'YOLOv8l': 'yolov8l.pt',
            'Custom-Plate-Detector': 'license_plate_detector.pt'
        }

        confidence_threshold = 0.3

        for model_name, model_path in detection_models.items():
            print(f"\n--- Testing {model_name} ---")

            try:
                # Skip if custom model doesn't exist
                if model_name == 'Custom-Plate-Detector' and not os.path.exists(model_path):
                    print(f"⚠ {model_path} not found. Download it using download_model.py")
                    continue

                # Load model
                model = YOLO(model_path)

                # Test metrics
                total_time = 0
                detections_count = 0
                confidences = []
                images_with_detection = 0

                for img_path in self.test_images:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Time the detection
                    start_time = time.time()
                    results = model(img, conf=confidence_threshold, verbose=False)
                    inference_time = time.time() - start_time

                    total_time += inference_time

                    # Count detections
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) > 0:
                            images_with_detection += 1
                        for box in boxes:
                            detections_count += 1
                            confidences.append(float(box.conf[0]))

                # Calculate metrics
                avg_time = total_time / len(self.test_images)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                detection_rate = (images_with_detection / len(self.test_images)) * 100

                # Store results
                self.results['detection_models'][model_name] = {
                    'avg_inference_time_ms': avg_time * 1000,
                    'total_detections': detections_count,
                    'avg_confidence': avg_confidence,
                    'detection_rate_percent': detection_rate,
                    'images_with_detection': images_with_detection,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0
                }

                print(f"✓ Avg Inference Time: {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)")
                print(f"✓ Detection Rate: {detection_rate:.1f}% ({images_with_detection}/{len(self.test_images)} images)")
                print(f"✓ Total Detections: {detections_count}")
                print(f"✓ Avg Confidence: {avg_confidence:.3f}")

            except Exception as e:
                print(f"✗ Error testing {model_name}: {e}")
                self.results['detection_models'][model_name] = {'error': str(e)}

        return self.results['detection_models']

    def test_ocr_models(self, use_detection_model='YOLOv8n'):
        """Test different OCR models on detected plates."""
        print("\n" + "="*60)
        print("TESTING OCR MODELS")
        print("="*60)

        # First detect plates with chosen model
        print(f"Using {use_detection_model} for plate detection...")

        try:
            # Load detection model
            if use_detection_model == 'Custom-Plate-Detector' and os.path.exists('license_plate_detector.pt'):
                detector = YOLO('license_plate_detector.pt')
            else:
                detector = YOLO('yolov8n.pt')

            # Detect plates from all test images
            detected_plates = []
            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                results = detector(img, conf=0.3, verbose=False)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Add padding
                        h_img, w_img = img.shape[:2]
                        box_w = x2 - x1
                        box_h = y2 - y1
                        pad_x = int(box_w * 0.2)
                        pad_y = int(box_h * 0.05)
                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(w_img, x2 + pad_x)
                        y2 = min(h_img, y2 + pad_y)

                        plate_img = img[y1:y2, x1:x2]
                        detected_plates.append({
                            'image': plate_img,
                            'source': img_path.name
                        })

            print(f"Detected {len(detected_plates)} plates for OCR testing")

            if len(detected_plates) == 0:
                print("⚠ No plates detected. Cannot test OCR models.")
                return {}

            # Test OCR models
            ocr_models = {}

            # 1. PaddleOCR
            print("\n--- Testing PaddleOCR ---")
            try:
                ocr_paddle = PaddleOCR(use_textline_orientation=True, lang='en')

                total_time = 0
                successful_reads = 0
                confidences = []
                extracted_texts = []

                for plate_data in detected_plates:
                    plate_img = plate_data['image']
                    if plate_img is None or len(plate_img) == 0:
                        continue

                    # Convert BGR to RGB
                    plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

                    start_time = time.time()
                    results = ocr_paddle.ocr(plate_rgb, cls=True)
                    inference_time = time.time() - start_time
                    total_time += inference_time

                    # Extract text
                    if results and results[0]:
                        text_parts = []
                        scores = []
                        for line in results[0]:
                            if isinstance(line, list) and len(line) == 2:
                                text_parts.append(line[1][0])
                                scores.append(line[1][1])

                        if text_parts:
                            successful_reads += 1
                            extracted_texts.append({
                                'source': plate_data['source'],
                                'text': ' '.join(text_parts),
                                'confidence': np.mean(scores)
                            })
                            confidences.append(np.mean(scores))

                ocr_models['PaddleOCR'] = {
                    'avg_inference_time_ms': (total_time / len(detected_plates)) * 1000,
                    'successful_reads': successful_reads,
                    'success_rate_percent': (successful_reads / len(detected_plates)) * 100,
                    'avg_confidence': np.mean(confidences) if confidences else 0.0,
                    'sample_texts': extracted_texts[:10]
                }

                print(f"✓ Avg Inference Time: {(total_time/len(detected_plates))*1000:.2f}ms")
                print(f"✓ Success Rate: {(successful_reads/len(detected_plates))*100:.1f}%")
                print(f"✓ Avg Confidence: {np.mean(confidences) if confidences else 0:.3f}")

            except Exception as e:
                print(f"✗ Error testing PaddleOCR: {e}")
                ocr_models['PaddleOCR'] = {'error': str(e)}

            # 2. EasyOCR
            print("\n--- Testing EasyOCR ---")
            try:
                import easyocr
                ocr_easy = easyocr.Reader(['en'], gpu=False)

                total_time = 0
                successful_reads = 0
                confidences = []
                extracted_texts = []

                for plate_data in detected_plates:
                    plate_img = plate_data['image']
                    if plate_img is None or len(plate_img) == 0:
                        continue

                    start_time = time.time()
                    results = ocr_easy.readtext(plate_img)
                    inference_time = time.time() - start_time
                    total_time += inference_time

                    if results:
                        texts = [text for (_, text, conf) in results]
                        confs = [conf for (_, text, conf) in results]

                        if texts:
                            successful_reads += 1
                            extracted_texts.append({
                                'source': plate_data['source'],
                                'text': ' '.join(texts),
                                'confidence': np.mean(confs)
                            })
                            confidences.append(np.mean(confs))

                ocr_models['EasyOCR'] = {
                    'avg_inference_time_ms': (total_time / len(detected_plates)) * 1000,
                    'successful_reads': successful_reads,
                    'success_rate_percent': (successful_reads / len(detected_plates)) * 100,
                    'avg_confidence': np.mean(confidences) if confidences else 0.0,
                    'sample_texts': extracted_texts[:10]
                }

                print(f"✓ Avg Inference Time: {(total_time/len(detected_plates))*1000:.2f}ms")
                print(f"✓ Success Rate: {(successful_reads/len(detected_plates))*100:.1f}%")
                print(f"✓ Avg Confidence: {np.mean(confidences) if confidences else 0:.3f}")

            except ImportError:
                print("⚠ EasyOCR not installed. Install with: pip install easyocr")
                ocr_models['EasyOCR'] = {'error': 'Not installed'}
            except Exception as e:
                print(f"✗ Error testing EasyOCR: {e}")
                ocr_models['EasyOCR'] = {'error': str(e)}

            # 3. Tesseract OCR
            print("\n--- Testing Tesseract OCR ---")
            try:
                import pytesseract

                total_time = 0
                successful_reads = 0
                extracted_texts = []

                for plate_data in detected_plates:
                    plate_img = plate_data['image']
                    if plate_img is None or len(plate_img) == 0:
                        continue

                    # Preprocess for Tesseract
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                    start_time = time.time()
                    text = pytesseract.image_to_string(gray, config='--psm 7')
                    inference_time = time.time() - start_time
                    total_time += inference_time

                    text = text.strip()
                    if text:
                        successful_reads += 1
                        extracted_texts.append({
                            'source': plate_data['source'],
                            'text': text
                        })

                ocr_models['Tesseract'] = {
                    'avg_inference_time_ms': (total_time / len(detected_plates)) * 1000,
                    'successful_reads': successful_reads,
                    'success_rate_percent': (successful_reads / len(detected_plates)) * 100,
                    'sample_texts': extracted_texts[:10]
                }

                print(f"✓ Avg Inference Time: {(total_time/len(detected_plates))*1000:.2f}ms")
                print(f"✓ Success Rate: {(successful_reads/len(detected_plates))*100:.1f}%")

            except ImportError:
                print("⚠ Tesseract not installed. Install with: pip install pytesseract")
                print("   Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract")
                ocr_models['Tesseract'] = {'error': 'Not installed'}
            except Exception as e:
                print(f"✗ Error testing Tesseract: {e}")
                ocr_models['Tesseract'] = {'error': str(e)}

            self.results['ocr_models'] = ocr_models
            return ocr_models

        except Exception as e:
            print(f"✗ Error in OCR testing: {e}")
            return {}

    def test_combined_pipelines(self):
        """Test end-to-end pipelines with different model combinations."""
        print("\n" + "="*60)
        print("TESTING COMBINED PIPELINES (Detection + OCR)")
        print("="*60)

        # Best combinations to test
        pipelines = [
            ('YOLOv8n', 'PaddleOCR', 'Fast & Balanced'),
            ('YOLOv8s', 'PaddleOCR', 'Better Detection'),
            ('Custom-Plate-Detector', 'PaddleOCR', 'Custom Model'),
        ]

        for det_model, ocr_model, description in pipelines:
            print(f"\n--- Testing: {det_model} + {ocr_model} ({description}) ---")

            try:
                # Load models
                if det_model == 'Custom-Plate-Detector' and not os.path.exists('license_plate_detector.pt'):
                    print(f"⚠ Custom model not found. Skipping.")
                    continue

                detector = YOLO('license_plate_detector.pt' if det_model == 'Custom-Plate-Detector' else f'{det_model.lower()}.pt')

                if ocr_model == 'PaddleOCR':
                    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
                else:
                    print(f"⚠ {ocr_model} not implemented yet")
                    continue

                # Test on all images
                total_time = 0
                successful_reads = 0
                all_results = []

                for img_path in self.test_images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    pipeline_start = time.time()

                    # Detect
                    det_results = detector(img, conf=0.3, verbose=False)

                    # OCR on each detection
                    for result in det_results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Padding
                            h_img, w_img = img.shape[:2]
                            box_w = x2 - x1
                            box_h = y2 - y1
                            pad_x = int(box_w * 0.2)
                            pad_y = int(box_h * 0.05)
                            x1 = max(0, x1 - pad_x)
                            y1 = max(0, y1 - pad_y)
                            x2 = min(w_img, x2 + pad_x)
                            y2 = min(h_img, y2 + pad_y)

                            plate_img = img[y1:y2, x1:x2]
                            plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

                            # OCR
                            ocr_results = ocr.ocr(plate_rgb, cls=True)

                            if ocr_results and ocr_results[0]:
                                texts = []
                                for line in ocr_results[0]:
                                    if isinstance(line, list) and len(line) == 2:
                                        texts.append(line[1][0])

                                if texts:
                                    successful_reads += 1
                                    all_results.append({
                                        'image': img_path.name,
                                        'text': ' '.join(texts),
                                        'det_conf': float(box.conf[0])
                                    })

                    pipeline_time = time.time() - pipeline_start
                    total_time += pipeline_time

                # Calculate metrics
                avg_time = total_time / len(self.test_images)
                success_rate = (successful_reads / len(self.test_images)) * 100

                pipeline_name = f"{det_model}+{ocr_model}"
                self.results['combined_results'][pipeline_name] = {
                    'description': description,
                    'avg_pipeline_time_ms': avg_time * 1000,
                    'successful_reads': successful_reads,
                    'success_rate_percent': success_rate,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0,
                    'sample_results': all_results[:15]
                }

                print(f"✓ Avg Pipeline Time: {avg_time*1000:.2f}ms ({1/avg_time:.1f} FPS)")
                print(f"✓ Success Rate: {success_rate:.1f}% ({successful_reads}/{len(self.test_images)} images)")

            except Exception as e:
                print(f"✗ Error testing pipeline: {e}")

    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)

        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"model_comparison_report_{timestamp}.json"

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_images_count': len(self.test_images),
            'test_data_path': str(self.test_data_path),
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n✓ JSON Report saved: {report_file}")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*60)

        # Best detection model
        if self.results['detection_models']:
            print("\n🔍 DETECTION MODELS:")
            det_sorted = sorted(
                [(k, v) for k, v in self.results['detection_models'].items() if 'error' not in v],
                key=lambda x: (x[1].get('detection_rate_percent', 0), -x[1].get('avg_inference_time_ms', float('inf'))),
                reverse=True
            )

            for i, (name, metrics) in enumerate(det_sorted[:3], 1):
                print(f"  {i}. {name}")
                print(f"     - Detection Rate: {metrics.get('detection_rate_percent', 0):.1f}%")
                print(f"     - Speed: {metrics.get('avg_inference_time_ms', 0):.2f}ms ({metrics.get('fps', 0):.1f} FPS)")
                print(f"     - Confidence: {metrics.get('avg_confidence', 0):.3f}")

        # Best OCR model
        if self.results['ocr_models']:
            print("\n📖 OCR MODELS:")
            ocr_sorted = sorted(
                [(k, v) for k, v in self.results['ocr_models'].items() if 'error' not in v],
                key=lambda x: (x[1].get('success_rate_percent', 0), -x[1].get('avg_inference_time_ms', float('inf'))),
                reverse=True
            )

            for i, (name, metrics) in enumerate(ocr_sorted[:3], 1):
                print(f"  {i}. {name}")
                print(f"     - Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
                print(f"     - Speed: {metrics.get('avg_inference_time_ms', 0):.2f}ms")
                if 'avg_confidence' in metrics:
                    print(f"     - Confidence: {metrics.get('avg_confidence', 0):.3f}")

        # Best combined pipeline
        combined_sorted = []
        if self.results['combined_results']:
            print("\n🚀 COMBINED PIPELINES:")
            combined_sorted = sorted(
                [(k, v) for k, v in self.results['combined_results'].items()],
                key=lambda x: (x[1].get('success_rate_percent', 0), -x[1].get('avg_pipeline_time_ms', float('inf'))),
                reverse=True
            )

            for i, (name, metrics) in enumerate(combined_sorted[:3], 1):
                print(f"  {i}. {name} - {metrics.get('description', '')}")
                print(f"     - Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
                print(f"     - Speed: {metrics.get('avg_pipeline_time_ms', 0):.2f}ms ({metrics.get('fps', 0):.1f} FPS)")

        # Recommendation
        print("\n💡 RECOMMENDATION:")
        if combined_sorted and len(combined_sorted) > 0:
            best = combined_sorted[0]
            print(f"   Use: {best[0]}")
            print(f"   Reason: Best overall performance with {best[1].get('success_rate_percent', 0):.1f}% success rate")
            print(f"   and {best[1].get('avg_pipeline_time_ms', 0):.2f}ms processing time")

        print("\n" + "="*60)

        return report_file


def main():
    """Main execution function."""
    print("="*60)
    print("VEHICLE REGISTRATION MODEL COMPARISON")
    print("="*60)

    # Test data path
    test_data_path = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_data_path):
        print(f"Error: Test data path not found: {test_data_path}")
        return

    # Initialize comparison
    comparison = ModelComparison(test_data_path)

    # Run tests
    print(f"\nStarting tests with {len(comparison.test_images)} images...")

    # 1. Test detection models
    comparison.test_detection_models()

    # 2. Test OCR models
    comparison.test_ocr_models()

    # 3. Test combined pipelines
    comparison.test_combined_pipelines()

    # 4. Generate report
    report_file = comparison.generate_report()

    print(f"\n✅ Testing complete! Check {report_file} for detailed results.")


if __name__ == "__main__":
    main()
