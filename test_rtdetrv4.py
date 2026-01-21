"""
RT-DETRv4 vs Custom YOLO Plate Detector Comparison
Tests the latest RT-DETRv4 model against the custom plate detector
"""

import os
import cv2
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Import existing OCR
import sys
sys.path.append(os.path.dirname(__file__))
from services.plate_ocr import PlateOCR

class RTDETRv4Tester:
    def __init__(self, test_folder):
        self.test_folder = test_folder
        self.ocr = PlateOCR()
        self.results = {}

    def test_rtdetrv4(self):
        """Test RT-DETRv4 model"""
        print("\n" + "="*60)
        print("Testing RT-DETRv4")
        print("="*60)

        try:
            from ultralytics import RTDETR

            # Try different RT-DETRv4 model paths
            model_paths = [
                'rtdetr-v4.pt',
                'rtdetrv4.pt',
                'rtdetr-x.pt',  # Latest/largest might be v4
                'rtdetr-l.pt'   # Fall back to large
            ]

            detector = None
            model_used = None

            for model_path in model_paths:
                try:
                    print(f"[INFO] Trying to load {model_path}...")
                    detector = RTDETR(model_path)
                    model_used = model_path
                    print(f"[SUCCESS] Loaded {model_path}")
                    break
                except Exception as e:
                    print(f"[SKIP] {model_path} not available: {str(e)[:50]}")
                    continue

            if detector is None:
                print("[ERROR] No RT-DETR model could be loaded")
                return None

            return self._test_detector(detector, f"RT-DETRv4 ({model_used})")

        except ImportError as e:
            print(f"[ERROR] RT-DETR not available: {e}")
            return None

    def test_custom_yolo(self):
        """Test Custom YOLO Plate Detector"""
        print("\n" + "="*60)
        print("Testing Custom YOLO Plate Detector")
        print("="*60)

        try:
            from ultralytics import YOLO

            model_path = 'license_plate_detector.pt'
            if not os.path.exists(model_path):
                print(f"[ERROR] {model_path} not found")
                return None

            detector = YOLO(model_path)
            return self._test_detector(detector, "Custom YOLO")

        except Exception as e:
            print(f"[ERROR] Failed to load custom model: {e}")
            return None

    def _test_detector(self, detector, model_name):
        """Run detection and OCR pipeline"""

        image_files = [f for f in os.listdir(self.test_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print("[ERROR] No images found in test folder")
            return None

        print(f"[INFO] Testing on {len(image_files)} images")

        all_detections = []
        successful_reads = []
        images_with_detection = set()
        det_times = []
        ocr_times = []
        det_confidences = []
        ocr_confidences = []

        for img_file in image_files:
            img_path = os.path.join(self.test_folder, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[SKIP] Could not read {img_file}")
                continue

            # Detection
            det_start = time.time()
            results = detector(img, verbose=False)
            det_time = (time.time() - det_start) * 1000
            det_times.append(det_time)

            # Process detections
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    conf = float(box.conf[0])

                    # Only process high confidence detections
                    if conf < 0.3:
                        continue

                    det_confidences.append(conf)
                    images_with_detection.add(img_file)

                    # Extract plate region
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_img = img[y1:y2, x1:x2]

                    if plate_img.size == 0:
                        continue

                    # OCR
                    ocr_start = time.time()
                    ocr_result = self.ocr.read_plate(plate_img)
                    ocr_time = (time.time() - ocr_start) * 1000
                    ocr_times.append(ocr_time)

                    text = ocr_result.get('text', '')
                    ocr_conf = ocr_result.get('confidence', 0.0)

                    all_detections.append({
                        'image': img_file,
                        'text': text,
                        'det_conf': conf,
                        'ocr_conf': ocr_conf,
                        'det_time': det_time,
                        'ocr_time': ocr_time
                    })

                    if text and len(text) >= 6:
                        successful_reads.append({
                            'image': img_file,
                            'text': text,
                            'det_conf': conf,
                            'ocr_conf': ocr_conf
                        })
                        ocr_confidences.append(ocr_conf)
                        print(f"[OK] {img_file}: {text} (det:{conf:.3f}, ocr:{ocr_conf:.3f})")
                    else:
                        print(f"[FAIL] {img_file}: '{text}' (too short/empty)")

        # Calculate metrics
        total_images = len(image_files)
        detection_rate = (len(images_with_detection) / total_images) * 100
        ocr_success_rate = (len(successful_reads) / len(all_detections) * 100) if all_detections else 0
        end_to_end_success = (len(successful_reads) / total_images) * 100

        avg_det_time = np.mean(det_times) if det_times else 0
        avg_ocr_time = np.mean(ocr_times) if ocr_times else 0
        avg_total_time = avg_det_time + avg_ocr_time

        avg_det_conf = np.mean(det_confidences) if det_confidences else 0
        avg_ocr_conf = np.mean(ocr_confidences) if ocr_confidences else 0

        result = {
            'model_name': model_name,
            'total_detections': len(all_detections),
            'successful_reads': len(successful_reads),
            'images_with_detection': len(images_with_detection),
            'total_images': total_images,
            'detection_rate': detection_rate,
            'ocr_success_rate': ocr_success_rate,
            'end_to_end_success': end_to_end_success,
            'avg_det_time_ms': avg_det_time,
            'avg_ocr_time_ms': avg_ocr_time,
            'avg_total_time_ms': avg_total_time,
            'avg_det_conf': avg_det_conf,
            'avg_ocr_conf': avg_ocr_conf,
            'sample_results': successful_reads[:10]
        }

        print(f"\n[SUMMARY] {model_name}")
        print(f"  Detection Rate: {detection_rate:.1f}% ({len(images_with_detection)}/{total_images})")
        print(f"  Total Detections: {len(all_detections)}")
        print(f"  Successful Reads: {len(successful_reads)}")
        print(f"  OCR Success Rate: {ocr_success_rate:.1f}%")
        print(f"  End-to-End Success: {end_to_end_success:.1f}%")
        print(f"  Avg Detection Time: {avg_det_time:.1f}ms")
        print(f"  Avg OCR Time: {avg_ocr_time:.1f}ms")
        print(f"  Avg Total Time: {avg_total_time:.1f}ms")
        print(f"  Avg Detection Confidence: {avg_det_conf:.3f}")
        print(f"  Avg OCR Confidence: {avg_ocr_conf:.3f}")

        return result

    def generate_comparison_report(self):
        """Generate comparison report"""

        if not self.results:
            print("\n[ERROR] No results to compare")
            return

        print("\n" + "="*60)
        print("COMPARISON REPORT: RT-DETRv4 vs Custom YOLO")
        print("="*60)

        # Find the models
        rtdetr_result = None
        custom_result = None

        for model_name, result in self.results.items():
            if 'RT-DETR' in model_name:
                rtdetr_result = result
            elif 'Custom' in model_name:
                custom_result = result

        if not rtdetr_result or not custom_result:
            print("[ERROR] Missing results for comparison")
            return

        # Compare metrics
        print("\n1. DETECTION RATE")
        print(f"   RT-DETRv4:    {rtdetr_result['detection_rate']:.1f}%")
        print(f"   Custom YOLO:  {custom_result['detection_rate']:.1f}%")
        winner = "RT-DETRv4" if rtdetr_result['detection_rate'] > custom_result['detection_rate'] else "Custom YOLO"
        if rtdetr_result['detection_rate'] == custom_result['detection_rate']:
            winner = "TIE"
        print(f"   Winner: {winner}")

        print("\n2. OCR SUCCESS RATE")
        print(f"   RT-DETRv4:    {rtdetr_result['ocr_success_rate']:.1f}%")
        print(f"   Custom YOLO:  {custom_result['ocr_success_rate']:.1f}%")
        winner = "RT-DETRv4" if rtdetr_result['ocr_success_rate'] > custom_result['ocr_success_rate'] else "Custom YOLO"
        print(f"   Winner: {winner}")

        print("\n3. PIPELINE SPEED")
        print(f"   RT-DETRv4:    {rtdetr_result['avg_total_time_ms']:.1f}ms")
        print(f"   Custom YOLO:  {custom_result['avg_total_time_ms']:.1f}ms")
        winner = "RT-DETRv4" if rtdetr_result['avg_total_time_ms'] < custom_result['avg_total_time_ms'] else "Custom YOLO"
        speed_diff = abs(rtdetr_result['avg_total_time_ms'] - custom_result['avg_total_time_ms'])
        faster_pct = (speed_diff / max(rtdetr_result['avg_total_time_ms'], custom_result['avg_total_time_ms'])) * 100
        print(f"   Winner: {winner} ({faster_pct:.1f}% faster)")

        print("\n4. END-TO-END SUCCESS")
        print(f"   RT-DETRv4:    {rtdetr_result['end_to_end_success']:.1f}%")
        print(f"   Custom YOLO:  {custom_result['end_to_end_success']:.1f}%")
        winner = "RT-DETRv4" if rtdetr_result['end_to_end_success'] > custom_result['end_to_end_success'] else "Custom YOLO"
        print(f"   Winner: {winner}")

        print("\n5. DETECTION CONFIDENCE")
        print(f"   RT-DETRv4:    {rtdetr_result['avg_det_conf']:.3f}")
        print(f"   Custom YOLO:  {custom_result['avg_det_conf']:.3f}")
        winner = "RT-DETRv4" if rtdetr_result['avg_det_conf'] > custom_result['avg_det_conf'] else "Custom YOLO"
        print(f"   Winner: {winner}")

        # Overall winner
        print("\n" + "="*60)
        print("OVERALL VERDICT")
        print("="*60)

        # Score based on key metrics
        rtdetr_score = 0
        custom_score = 0

        if rtdetr_result['ocr_success_rate'] > custom_result['ocr_success_rate']:
            rtdetr_score += 2  # OCR is most important
        else:
            custom_score += 2

        if rtdetr_result['avg_total_time_ms'] < custom_result['avg_total_time_ms']:
            rtdetr_score += 1  # Speed is important
        else:
            custom_score += 1

        if rtdetr_result['detection_rate'] > custom_result['detection_rate']:
            rtdetr_score += 1
        else:
            custom_score += 1

        if rtdetr_result['end_to_end_success'] > custom_result['end_to_end_success']:
            rtdetr_score += 1
        else:
            custom_score += 1

        if rtdetr_score > custom_score:
            print("\nWINNER: RT-DETRv4")
            print(f"Score: RT-DETRv4 ({rtdetr_score}) vs Custom YOLO ({custom_score})")
        elif custom_score > rtdetr_score:
            print("\nWINNER: Custom YOLO Plate Detector")
            print(f"Score: Custom YOLO ({custom_score}) vs RT-DETRv4 ({rtdetr_score})")
        else:
            print("\nRESULT: TIE")
            print(f"Score: Both models scored {rtdetr_score}")

        # Save JSON report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'rtdetrv4_comparison_{timestamp}.json'

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_folder': self.test_folder,
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n[SAVED] Report saved to {report_file}")

    def run_comparison(self):
        """Run complete comparison"""
        print("\n" + "="*60)
        print("RT-DETRv4 vs Custom YOLO - License Plate Detection")
        print("="*60)
        print(f"Test Folder: {self.test_folder}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Test RT-DETRv4
        rtdetr_result = self.test_rtdetrv4()
        if rtdetr_result:
            self.results[rtdetr_result['model_name']] = rtdetr_result

        # Test Custom YOLO
        custom_result = self.test_custom_yolo()
        if custom_result:
            self.results[custom_result['model_name']] = custom_result

        # Generate comparison
        if len(self.results) >= 2:
            self.generate_comparison_report()
        else:
            print("\n[ERROR] Could not run comparison - insufficient results")

def main():
    test_folder = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_folder):
        print(f"[ERROR] Test folder not found: {test_folder}")
        return

    tester = RTDETRv4Tester(test_folder)
    tester.run_comparison()

if __name__ == "__main__":
    main()
