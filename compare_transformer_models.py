"""
State-of-the-Art Transformer-Based Object Detection Comparison
Test against cutting-edge models:
- DETR (DEtection TRansformer)
- DINO (DETR with Improved deNoising)
- RT-DETR (Real-Time DETR)
- Deformable DETR
- YOLOv10
- YOLO-World
- SAM (Segment Anything)
- GroundingDINO
"""

import os
import sys
import cv2
import time
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from services.plate_ocr import PlateOCR


class TransformerModelComparison:
    """Compare transformer-based detection models."""

    def __init__(self, test_data_path):
        self.test_data_path = Path(test_data_path)
        self.ocr = PlateOCR()
        self.test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.test_images.extend(list(self.test_data_path.glob(ext)))
        print(f"Found {len(self.test_images)} test images")
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def test_custom_yolo(self):
        """Baseline: Custom YOLO plate detector."""
        print(f"\n{'='*60}")
        print(f"Testing: Custom YOLO Plate Detector (Baseline)")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLO
            detector = YOLO('license_plate_detector.pt')

            metrics = self._run_pipeline(
                'Custom YOLO',
                lambda img: self._yolo_detect(detector, img)
            )
            return metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def test_rt_detr(self):
        """Test RT-DETR (Real-Time DETR from Ultralytics)."""
        print(f"\n{'='*60}")
        print(f"Testing: RT-DETR (Real-Time Transformer)")
        print(f"{'='*60}")

        try:
            from ultralytics import RTDETR

            # Try different RT-DETR models
            model_variants = [
                ('rtdetr-l', 'RT-DETR-L'),
                ('rtdetr-x', 'RT-DETR-X'),
            ]

            for model_name, display_name in model_variants:
                try:
                    print(f"\n--- Testing {display_name} ---")
                    detector = RTDETR(f'{model_name}.pt')

                    metrics = self._run_pipeline(
                        display_name,
                        lambda img: self._rtdetr_detect(detector, img)
                    )

                    if metrics:
                        break  # Use first successful model

                except Exception as e:
                    print(f"[SKIP] {display_name}: {e}")
                    continue

            return metrics if 'metrics' in locals() else None

        except ImportError:
            print("[SKIP] RT-DETR not available. Update ultralytics: pip install -U ultralytics")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def test_yolov10(self):
        """Test YOLOv10 (latest YOLO with NMS-free)."""
        print(f"\n{'='*60}")
        print(f"Testing: YOLOv10 (NMS-free Architecture)")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLO

            # YOLOv10 variants
            model_variants = [
                ('yolov10n.pt', 'YOLOv10n'),
                ('yolov10s.pt', 'YOLOv10s'),
                ('yolov10m.pt', 'YOLOv10m'),
            ]

            best_metrics = None

            for model_path, display_name in model_variants:
                try:
                    print(f"\n--- Testing {display_name} ---")
                    detector = YOLO(model_path)

                    metrics = self._run_pipeline(
                        display_name,
                        lambda img: self._yolo_detect(detector, img)
                    )

                    if metrics and (best_metrics is None or
                                  metrics['end_to_end_success'] > best_metrics['end_to_end_success']):
                        best_metrics = metrics

                except Exception as e:
                    print(f"[SKIP] {display_name}: {e}")
                    continue

            return best_metrics

        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def test_detr(self):
        """Test DETR (original Detection Transformer)."""
        print(f"\n{'='*60}")
        print(f"Testing: DETR (Detection Transformer)")
        print(f"{'='*60}")

        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            from PIL import Image

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            model.to(self.device)

            metrics = self._run_pipeline(
                'DETR',
                lambda img: self._detr_detect(model, processor, img)
            )
            return metrics

        except ImportError:
            print("[SKIP] Transformers not installed. Install: pip install transformers")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_dino(self):
        """Test DINO (DETR with Improved deNoising)."""
        print(f"\n{'='*60}")
        print(f"Testing: DINO (Improved DETR)")
        print(f"{'='*60}")

        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            from PIL import Image

            processor = AutoImageProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
            model.to(self.device)

            metrics = self._run_pipeline(
                'DINO',
                lambda img: self._dino_detect(model, processor, img)
            )
            return metrics

        except Exception as e:
            print(f"[SKIP] DINO not available: {e}")
            return None

    def test_deformable_detr(self):
        """Test Deformable DETR."""
        print(f"\n{'='*60}")
        print(f"Testing: Deformable DETR")
        print(f"{'='*60}")

        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
            model = AutoModelForObjectDetection.from_pretrained("SenseTime/deformable-detr")
            model.to(self.device)

            metrics = self._run_pipeline(
                'Deformable DETR',
                lambda img: self._detr_detect(model, processor, img)
            )
            return metrics

        except Exception as e:
            print(f"[SKIP] Deformable DETR not available: {e}")
            return None

    def test_yolo_world(self):
        """Test YOLO-World (open-vocabulary detection)."""
        print(f"\n{'='*60}")
        print(f"Testing: YOLO-World (Open-Vocabulary)")
        print(f"{'='*60}")

        try:
            from ultralytics import YOLOWorld

            detector = YOLOWorld("yolov8s-world.pt")

            # Set custom classes for plate detection
            detector.set_classes(["license plate", "number plate", "car plate", "vehicle plate"])

            metrics = self._run_pipeline(
                'YOLO-World',
                lambda img: self._yolo_world_detect(detector, img)
            )
            return metrics

        except ImportError:
            print("[SKIP] YOLO-World not available. Update ultralytics: pip install -U ultralytics")
            return None
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def _yolo_detect(self, detector, img):
        """Standard YOLO detection."""
        results = detector(img, conf=0.3, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

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

    def _rtdetr_detect(self, detector, img):
        """RT-DETR detection."""
        results = detector(img, conf=0.3, verbose=False)
        detections = []

        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

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

    def _detr_detect(self, model, processor, img):
        """DETR-based detection."""
        from PIL import Image

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([img.shape[:2]]).to(self.device)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.3
        )[0]

        detections = []

        # Filter for vehicle-related classes (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
        vehicle_classes = [2, 3, 5, 7]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() in vehicle_classes and score.item() > 0.3:
                x1, y1, x2, y2 = map(int, box.tolist())

                # Extract plate region (lower portion of vehicle)
                h_box = y2 - y1
                y1_plate = int(y1 + h_box * 0.6)

                detections.append({
                    'box': (x1, y1_plate, x2, y2),
                    'confidence': float(score.item())
                })

        return detections

    def _dino_detect(self, model, processor, img):
        """DINO detection."""
        return self._detr_detect(model, processor, img)

    def _yolo_world_detect(self, detector, img):
        """YOLO-World open-vocabulary detection."""
        results = detector.predict(img, conf=0.3, verbose=False)
        detections = []

        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

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

    def _run_pipeline(self, model_name, detect_func):
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

        num_images = len(self.test_images)
        metrics = {
            'model_name': model_name,
            'total_detections': total_detections,
            'successful_reads': successful_reads,
            'images_with_detection': images_with_detection,
            'detection_rate': (images_with_detection / num_images) * 100,
            'avg_det_time_ms': (total_time_det / num_images) * 1000,
            'avg_ocr_time_ms': (total_time_ocr / total_detections) * 1000 if total_detections > 0 else 0,
            'avg_total_time_ms': ((total_time_det / num_images) + (total_time_ocr / total_detections if total_detections > 0 else 0)) * 1000,
            'avg_det_conf': float(np.mean(det_confs)) if det_confs else 0,
            'avg_ocr_conf': float(np.mean(ocr_confs)) if ocr_confs else 0,
            'ocr_success_rate': (successful_reads / total_detections) * 100 if total_detections > 0 else 0,
            'end_to_end_success': (successful_reads / num_images) * 100,
            'sample_results': results[:10]
        }

        print(f"\nSummary:")
        print(f"  Detection Rate: {metrics['detection_rate']:.1f}% ({images_with_detection}/{num_images})")
        print(f"  OCR Success: {metrics['ocr_success_rate']:.1f}% ({successful_reads}/{total_detections})")
        print(f"  End-to-End: {metrics['end_to_end_success']:.1f}%")
        print(f"  Speed: {metrics['avg_total_time_ms']:.1f}ms")

        self.results[model_name] = metrics
        return metrics

    def generate_report(self):
        """Generate comparison report."""
        if not self.results:
            print("No results to compare")
            return

        print("\n" + "="*60)
        print("TRANSFORMER MODELS COMPARISON REPORT")
        print("="*60)

        # Sort by end-to-end success
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('end_to_end_success', 0),
            reverse=True
        )

        print("\n1. END-TO-END SUCCESS:")
        print("-"*60)
        for i, (name, m) in enumerate(sorted_results, 1):
            success = m.get('end_to_end_success', 0)
            time_ms = m.get('avg_total_time_ms', 0)
            print(f"{i}. {name:25s} {success:6.1f}% | {time_ms:7.1f}ms")

        print("\n2. DETECTION RATE:")
        print("-"*60)
        sorted_det = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('detection_rate', 0),
            reverse=True
        )
        for i, (name, m) in enumerate(sorted_det, 1):
            rate = m.get('detection_rate', 0)
            print(f"{i}. {name:25s} {rate:6.1f}%")

        print("\n3. OCR SUCCESS RATE:")
        print("-"*60)
        sorted_ocr = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('ocr_success_rate', 0),
            reverse=True
        )
        for i, (name, m) in enumerate(sorted_ocr, 1):
            rate = m.get('ocr_success_rate', 0)
            print(f"{i}. {name:25s} {rate:6.1f}%")

        print("\n4. SPEED (Detection only):")
        print("-"*60)
        sorted_speed = sorted(
            [(k, v) for k, v in self.results.items()],
            key=lambda x: x[1].get('avg_det_time_ms', float('inf'))
        )
        for i, (name, m) in enumerate(sorted_speed, 1):
            time_ms = m.get('avg_det_time_ms', 0)
            print(f"{i}. {name:25s} {time_ms:7.1f}ms")

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"transformer_comparison_{timestamp}.json"

        report_data = {
            'test_date': datetime.now().isoformat(),
            'test_images': len(self.test_images),
            'device': self.device,
            'results': self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved: {report_file}")
        print("="*60)


def main():
    """Main execution."""
    print("="*60)
    print("STATE-OF-THE-ART TRANSFORMER MODEL COMPARISON")
    print("="*60)

    test_path = r"C:\Users\Rishichowdary-3925\Downloads\No plates"

    if not os.path.exists(test_path):
        print(f"[ERROR] Path not found: {test_path}")
        return

    comparison = TransformerModelComparison(test_path)

    # Test all models
    comparison.test_custom_yolo()       # Baseline
    comparison.test_yolov10()           # Latest YOLO
    comparison.test_rt_detr()           # Real-Time Transformer
    comparison.test_yolo_world()        # Open-vocabulary
    comparison.test_detr()              # Original transformer
    comparison.test_deformable_detr()   # Deformable attention
    comparison.test_dino()              # Improved DETR

    # Generate report
    comparison.generate_report()


if __name__ == "__main__":
    main()
