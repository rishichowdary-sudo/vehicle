"""
Detection Model Comparison - Test different YOLO models vs Custom Plate Detector
Uses your existing OCR implementation to compare end-to-end accuracy
"""

import os
import sys
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

# Import your existing services
sys.path.append(os.path.dirname(__file__))
from services.plate_ocr import PlateOCR


class DetectionModelComparison:
    """Compare different detection models with your existing OCR."""

    def __init__(self, test_data_path):
        """Initialize with test data path."""
        self.test_data_path = Path(test_data_path)
        self.ocr = PlateOCR()  # Your existing OCR

        # Get all test images
        self.test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.test_images.extend(list(self.test_data_path.glob(ext)))

        print(f"Found {len(self.test_images)} test images")
        print(f"Using your existing PaddleOCR implementation")

        self.results = {}

    def test_detection_model(self, model_name, model_path, confidence_threshold=0.3):
        """Test a single detection model end-to-end with OCR."""
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")

        try:
            # Check if model exists for custom model
            if 'Custom' in model_name and not os.path.exists(model_path):
                print(f"[ERROR] {model_path} not found!")
                return None

            # Load detection model
            detector = YOLO(model_path)

            # Metrics
            total_time_detection = 0
            total_time_ocr = 0
            total_detections = 0
            successful_ocr_reads = 0
            detection_confidences = []
            ocr_confidences = []
            all_results = []
            images_with_plates = 0

            # Test on each image
            for img_path in self.test_images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # DETECTION
                start_det = time.time()
                det_results = detector(img, conf=confidence_threshold, verbose=False)
                detection_time = time.time() - start_det
                total_time_detection += detection_time

                image_had_detection = False

                # Process each detection
                for result in det_results:
                    boxes = result.boxes

                    for box in boxes:
                        total_detections += 1
                        image_had_detection = True

                        # Extract plate region
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        det_conf = float(box.conf[0])
                        detection_confidences.append(det_conf)

                        # Add padding (same as your system)
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

                        # OCR with your existing implementation
                        start_ocr = time.time()
                        ocr_result = self.ocr.read_plate(plate_img)
                        ocr_time = time.time() - start_ocr
                        total_time_ocr += ocr_time

                        plate_text = ocr_result.get('text', '')
                        ocr_conf = ocr_result.get('confidence', 0.0)

                        if plate_text and len(plate_text) > 3:  # Valid plate text
                            successful_ocr_reads += 1
                            ocr_confidences.append(ocr_conf)

                            all_results.append({
                                'image': img_path.name,
                                'plate_text': plate_text,
                                'det_confidence': det_conf,
                                'ocr_confidence': ocr_conf,
                                'det_time_ms': detection_time * 1000,
                                'ocr_time_ms': ocr_time * 1000
                            })

                            print(f"  [OK] {img_path.name}: '{plate_text}' (Det: {det_conf:.2f}, OCR: {ocr_conf:.2f})")
                        else:
                            print(f"  [FAIL] {img_path.name}: OCR failed (Det: {det_conf:.2f})")

                if image_had_detection:
                    images_with_plates += 1

            # Calculate metrics
            avg_det_time = (total_time_detection / len(self.test_images)) * 1000
            avg_ocr_time = (total_time_ocr / total_detections) * 1000 if total_detections > 0 else 0
            avg_total_time = avg_det_time + avg_ocr_time

            detection_rate = (images_with_plates / len(self.test_images)) * 100
            ocr_success_rate = (successful_ocr_reads / total_detections) * 100 if total_detections > 0 else 0
            end_to_end_success = (successful_ocr_reads / len(self.test_images)) * 100

            avg_det_conf = sum(detection_confidences) / len(detection_confidences) if detection_confidences else 0
            avg_ocr_conf = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0

            metrics = {
                'model_name': model_name,
                'model_path': model_path,
                # Detection metrics
                'detection_rate_percent': detection_rate,
                'total_detections': total_detections,
                'images_with_plates': images_with_plates,
                'avg_detection_confidence': avg_det_conf,
                'avg_detection_time_ms': avg_det_time,
                'detection_fps': 1000 / avg_det_time if avg_det_time > 0 else 0,
                # OCR metrics
                'ocr_success_rate_percent': ocr_success_rate,
                'successful_ocr_reads': successful_ocr_reads,
                'avg_ocr_confidence': avg_ocr_conf,
                'avg_ocr_time_ms': avg_ocr_time,
                # End-to-end metrics
                'end_to_end_success_rate': end_to_end_success,
                'avg_total_time_ms': avg_total_time,
                'total_fps': 1000 / avg_total_time if avg_total_time > 0 else 0,
                # Sample results
                'sample_results': all_results[:10]
            }

            # Print summary
            print(f"\n--- {model_name} Summary ---")
            print(f"Detection Rate: {detection_rate:.1f}% ({images_with_plates}/{len(self.test_images)} images)")
            print(f"Total Plates Detected: {total_detections}")
            print(f"Avg Detection Confidence: {avg_det_conf:.3f}")
            print(f"Detection Speed: {avg_det_time:.2f}ms ({1000/avg_det_time:.1f} FPS)")
            print(f"\nOCR Success Rate: {ocr_success_rate:.1f}% ({successful_ocr_reads}/{total_detections} plates)")
            print(f"Avg OCR Confidence: {avg_ocr_conf:.3f}")
            print(f"OCR Speed: {avg_ocr_time:.2f}ms per plate")
            print(f"\nEND-TO-END SUCCESS: {end_to_end_success:.1f}% ({successful_ocr_reads}/{len(self.test_images)} images)")
            print(f"Total Pipeline Time: {avg_total_time:.2f}ms ({1000/avg_total_time:.1f} FPS)")

            self.results[model_name] = metrics
            return metrics

        except Exception as e:
            print(f"[ERROR] Testing {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_tests(self):
        """Test all detection models."""
        print("\n" + "="*60)
        print("DETECTION MODEL COMPARISON")
        print("Testing different detection models with your existing OCR")
        print("="*60)

        # Models to test
        models = [
            ('YOLOv8n (Nano)', 'yolov8n.pt'),
            ('YOLOv8s (Small)', 'yolov8s.pt'),
            ('YOLOv8m (Medium)', 'yolov8m.pt'),
            ('YOLOv8l (Large)', 'yolov8l.pt'),
            ('Custom-Plate-Detector', 'license_plate_detector.pt'),
        ]

        for model_name, model_path in models:
            self.test_detection_model(model_name, model_path)

        return self.results

    def generate_comparison_report(self):
        """Generate final comparison report."""
        if not self.results:
            print("No results to compare!")
            return

        print("\n" + "="*60)
        print("FINAL COMPARISON REPORT")
        print("="*60)

        # Sort by end-to-end success rate
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1]['end_to_end_success_rate'],
            reverse=True
        )

        print("\n1. END-TO-END SUCCESS RATE (Detection + OCR):")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_results, 1):
            success = metrics['end_to_end_success_rate']
            total_time = metrics['avg_total_time_ms']
            fps = metrics['total_fps']
            print(f"{i}. {name:25s} {success:5.1f}% | {total_time:6.1f}ms | {fps:4.1f} FPS")

        # Sort by detection confidence
        sorted_det_conf = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1]['avg_detection_confidence'],
            reverse=True
        )

        print("\n2. DETECTION CONFIDENCE:")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_det_conf, 1):
            conf = metrics['avg_detection_confidence']
            rate = metrics['detection_rate_percent']
            print(f"{i}. {name:25s} {conf:.3f} | Detection Rate: {rate:.1f}%")

        # Sort by speed
        sorted_speed = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1]['avg_total_time_ms']
        )

        print("\n3. SPEED (Total Pipeline Time):")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_speed, 1):
            total = metrics['avg_total_time_ms']
            det = metrics['avg_detection_time_ms']
            ocr = metrics['avg_ocr_time_ms']
            fps = metrics['total_fps']
            print(f"{i}. {name:25s} {total:6.1f}ms | Det:{det:5.1f}ms + OCR:{ocr:5.1f}ms | {fps:4.1f}FPS")

        # Sort by OCR confidence
        sorted_ocr = sorted(
            [(k, v) for k, v in self.results.items() if v['avg_ocr_confidence'] > 0],
            key=lambda x: x[1]['avg_ocr_confidence'],
            reverse=True
        )

        print("\n4. OCR CONFIDENCE (on successfully read plates):")
        print("-" * 60)
        for i, (name, metrics) in enumerate(sorted_ocr, 1):
            conf = metrics['avg_ocr_confidence']
            success = metrics['ocr_success_rate_percent']
            print(f"{i}. {name:25s} {conf:.3f} | OCR Success: {success:.1f}%")

        # Recommendation
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)

        best_accuracy = sorted_results[0]
        best_speed = sorted_speed[0]
        best_det_conf = sorted_det_conf[0]

        print(f"\nBest Overall Accuracy: {best_accuracy[0]}")
        print(f"  - End-to-end success: {best_accuracy[1]['end_to_end_success_rate']:.1f}%")
        print(f"  - Speed: {best_accuracy[1]['avg_total_time_ms']:.1f}ms")

        print(f"\nFastest Model: {best_speed[0]}")
        print(f"  - Speed: {best_speed[1]['avg_total_time_ms']:.1f}ms ({best_speed[1]['total_fps']:.1f} FPS)")
        print(f"  - End-to-end success: {best_speed[1]['end_to_end_success_rate']:.1f}%")

        print(f"\nHighest Detection Confidence: {best_det_conf[0]}")
        print(f"  - Detection confidence: {best_det_conf[1]['avg_detection_confidence']:.3f}")
        print(f"  - End-to-end success: {best_det_conf[1]['end_to_end_success_rate']:.1f}%")

        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"detection_comparison_report_{timestamp}.json"

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_images_count': len(self.test_images),
            'test_data_path': str(self.test_data_path),
            'ocr_model': 'PaddleOCR (Your existing implementation)',
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nDetailed JSON report saved: {report_file}")
        print("="*60)

        return report_file


def main():
    """Main execution."""
    print("="*60)
    print("DETECTION MODEL COMPARISON TEST")
    print("Comparing YOLO models vs Custom Plate Detector")
    print("Using your existing PaddleOCR implementation")
    print("="*60)

    # Test data path
    test_data_path = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test data path not found: {test_data_path}")
        return

    # Run comparison
    comparison = DetectionModelComparison(test_data_path)
    comparison.run_all_tests()
    comparison.generate_comparison_report()

    print("\nTesting complete!")


if __name__ == "__main__":
    main()
